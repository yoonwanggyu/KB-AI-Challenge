import os
import json
import torch
import argparse
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.output_parsers import StrOutputParser


# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU to use

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 금융 상품의 유형을 결정하는 함수
def determine_product_type(key):
    if 'Housing' in key:
        return 'House'
    elif 'deposit' in key or 'Deposit' in key:
        return 'Deposit'
    elif 'savings' in key or 'Savings' in key:
        return 'Savings'
    elif 'Account' in key or 'account' in key:
        return 'Account'
    elif 'Loan' in key or 'loan' in key:
        return 'Loan'
    elif 'Check Card' in key:
        return 'Card'
    elif 'Card' in key or 'card' in key:
        return 'Card'
    else:
        return 'persona'

# 금융상품 DB, 사용자 DB에서 데이터 로드
def load_documents(base_dir, persona_pth):
    documents = []
    
    with open(persona_pth, 'r') as f:
        persona_doc = json.load(f)

    for key, value in persona_doc.items():
        product_type = determine_product_type(key)
        content = ''
        for sub_key, sub_value in value.items():
            content += f"{sub_key}: {sub_value}\n"
        documents.append(Document(page_content=content, metadata={'product_type': product_type, 'product_name': key}))

    for json_file in os.listdir(base_dir):
        with open(os.path.join(base_dir, json_file), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for key, value in json_data.items():
            product_type = determine_product_type(key)
            content = f"Product Type: {product_type}\nProduct Name: {key}\n{value}"
            documents.append(Document(page_content=content, metadata={'product_type': product_type, 'product_name': key}))

    return documents

# 임베딩 후 벡터 스토어에 저장
def create_embeddings_and_db(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True},
    )

    knowledge_vector_database = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    return knowledge_vector_database

# user의 소비 내역 정보와 보유하고 있는 카드의 혜택으로 최종 카드 추천에 사용할 정보를 생성하는 chain
def get_information_chain(knowledge_vector_database, react_information_example, user, card_name, recommend_query):
    llm = ChatOllama(model='gemma2', temperature=0.25, seed=42)

    prompt_react_information = PromptTemplate(
        input_variables=['question', 'react'],
        template='question:{question}\n react:\n{react}'
    )

    system_instruction = "You're a financial AI that's good at organizing what you're given"

    fewshot_prompt_information = FewShotPromptTemplate(
        examples=react_information_example,
        example_prompt=prompt_react_information,
        prefix=system_instruction,
        suffix='user_information:{user_information}\n user_own_card_benefits:{user_own_card_benefits}\n question:{question}\n react:\n',
        input_variables=['question', 'user_own_card_benefits', 'user_information']
    )

    information_chain = (
        fewshot_prompt_information
        | llm
        | StrOutputParser()
    )

    retriever_1 = knowledge_vector_database.as_retriever(search_kwargs={"k": 1})
    retriever_2 = knowledge_vector_database.as_retriever(search_kwargs={'filter': {'product_name': card_name}, 'k': 1})

    context_1 = retriever_1.invoke(user)[0]
    context_2 = retriever_2.invoke(card_name)[0]

    information_response = information_chain.invoke({'user_information': context_1, 'user_own_card_benefits': context_2, 'question': recommend_query})

    return information_response

# user에 대한 정보를 바탕으로 vector db 기반 retriever를 통해 정보와 유사한 카드를 추출하고 reranking을 통해 최종 카드 추천
def get_recommendation_chain(knowledge_vector_database, recommend_card_example, information_response, card_name):
    information_response = '\n'.join(information_response.split('\n')[1:-2])

    prompt_recommend_card = PromptTemplate(
        input_variables=['question', 'answer'],
        template='question:{question}\n answer:\n{answer}'
    )

    system_instruction_recommend = "You're a financial ai who recommends cards for logical reasons."

    fewshot_prompt_recommend = FewShotPromptTemplate(
        examples=recommend_card_example,
        example_prompt=prompt_recommend_card,
        prefix=system_instruction_recommend,
        suffix='question:{question}\ncontext:{context}answer:\n',
        input_variables=['question', 'context']
    )

    llm = ChatOllama(model='gemma2', temperature=0.1, seed=215)

    recommend_chain = (
        fewshot_prompt_recommend
        | llm
        | StrOutputParser()
    )

    retriever = knowledge_vector_database.as_retriever(search_kwargs={'filter': {'product_type': 'Card'}, 'k': 10})
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    

    recommend_query = f"""Based on below information:{information_response}.
    Recommend the best card Consider the following points:1. Exclude cards {card_name} 2.Explain the reasons for your recommendation by relating the user's spending patterns by category to the recommended card. 3.Explain the reasons for your recommendation by relating the benefits of the card the user currently holds to the recommended card. 4.Write which card you recommend at the very beginning."""

    compressed_docs = compression_retriever.invoke(recommend_query)
    
    recommend_response = recommend_chain.invoke({"question": recommend_query, "context": compressed_docs})

    return recommend_response

# json 파일을 로드 하는 함수
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--username', type=str, required=True, help="유저 이름")
    args = parser.parse_args()
    base_dir = './data_files/translated_json'
    persona_pth = './data_files/persona.json'

    with open(persona_pth, 'r') as f:
        persona_doc = json.load(f)

    documents = load_documents(base_dir, persona_pth)
    knowledge_vector_database = create_embeddings_and_db(documents)

    react_information_example = load_json_file('./data_files/example_prompts/react_information_example.json')
    recommend_card_example = load_json_file('./data_files/example_prompts/recommend_card_example.json')


    user = args.username
    card_name = persona_doc[user]['Owned Card']

    recommend_query = f"""Categorize {user}'s consumption pattern by sector. [sector]\n [date: amount] and Analyze all the benefits of the ppyroducts for {user} owned card."""

    information_response = get_information_chain(knowledge_vector_database, react_information_example, user, card_name, recommend_query)
    recommendation = get_recommendation_chain(knowledge_vector_database, recommend_card_example, information_response, card_name)

    print(recommendation)

if __name__ == "__main__":
    main()
