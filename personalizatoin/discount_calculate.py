import os
import json
import argparse
import torch
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama


# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# react 방식으로 소비 내역과 추천 카드의 혜택을 비교하여 추천카드를 사용했을 시 할인 받을 수 있는 금액을 계산하는 chain
def create_calculate_discount_chain(discount_react_examples):
    example_prompt_calculate_discount = PromptTemplate(
        input_variables=['question', 'react'],
        template='question:{question}\nreact:{react}'
    )

    system_instruction = "You're a financial AI that calculates discounts on spending."

    fewshot_prompt_calculate_discount = FewShotPromptTemplate(
        examples=discount_react_examples,
        example_prompt=example_prompt_calculate_discount,
        prefix=system_instruction,
        suffix='question:{question}\nuser_consumption_history:\n{context_1}\ncard_benefits:{context_2}\nreact:',
        input_variables=['question', 'context_1', 'context_2']
    )

    llm = ChatOllama(model='gemma2', seed=102)

    calculate_discount_chain = (
        fewshot_prompt_calculate_discount
        | llm
        | StrOutputParser()
    )

    return calculate_discount_chain


def retrieve_context(knowledge_vector_database, user, recommend_card_name):
    retriever_1 = knowledge_vector_database.as_retriever(search_kwargs={"k": 1})
    retriever_2 = knowledge_vector_database.as_retriever(search_kwargs={'filter': {'product_name': recommend_card_name}, 'k': 1})

    context_1 = retriever_1.invoke(user)
    context_2 = retriever_2.invoke(recommend_card_name)

    return context_1, context_2

# json 파일을 로드 하는 함수
def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    

def main():
    parser = argparse.ArgumentParser(description="")

    # 매개변수 추가
    parser.add_argument('--username', type=str, required=True, help="유저 이름")
    parser.add_argument('--recommendation_card', type=str, required=True, help="추천하는 카드")
    
    # 매개변수 파싱
    args = parser.parse_args()
    base_dir = './data_files/translated_json'
    persona_pth = './data_files/persona.json'

    documents = load_documents(base_dir, persona_pth)
    knowledge_vector_database = create_embeddings_and_db(documents)


    discount_react_examples = load_json_file('./data_files/example_prompts/discount_react_examples.json')

    calculate_discount_chain = create_calculate_discount_chain(discount_react_examples)

    user = args.username
    recommend_card_name = args.recommendation_card

    context_1, context_2 = retrieve_context(knowledge_vector_database, user, recommend_card_name)

    discount_query = f""""Provide a detailed analysis of [calculate discount amount]. Follow the process of thinking through the steps, performing specific actions, and making observations based on those actions. Structure your response as follows:

    Thought: Describe your initial thoughts on how to approach the problem or task.
    Action: Outline the actions you will take to gather the necessary information or perform the required calculations.
    Observation: Present the observations and results from those actions.
    Once you have gathered all the information, provide a summary of the final result or conclusion. Do not include any information or context beyond the requested steps."

    Example:

    thought1: I need to categorize [{user}'s consumption pattern] by [sector].
    action1: Categorize [{user}'s consumption history] by [sector] with dates and amount.
    observation1: [Detailed categorization result].

    thought2: I need to List [{recommend_card_name}'s benefits].
    action2: List all the [{recommend_card_name}'s benefits] .
    observation2: [Detailed analysis result].

    thought3: I need to combine all the information and calculate [exact discount amount for all consumption history in each category using the {recommend_card_name}].
    action3: Calculate and show the total expenditure and total discount amount for each sector.
    observation3: [Final calculated result].

    Summary: [Sum up the all discount amount]."""

    discount_response = calculate_discount_chain.invoke({'question': discount_query, 'context_1': context_1, 'context_2': context_2})
    print(discount_response)


if __name__ == "__main__":
    main()
