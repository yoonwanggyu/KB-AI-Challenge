import os
import json
import torch
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#데이터에 따라 product type 분류
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
        return 'Check Card'
    elif 'Card' in key or 'card' in key:
        return 'Credit Card'

#금융 DB에서 데이터 로드
def load_documents(base_dir):
    documents = []
    for json_file in os.listdir(base_dir):
        with open(os.path.join(base_dir, json_file), 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for key, value in json_data.items():
            product_type = determine_product_type(key)
            content = f"Product Type: {product_type}\nProduct Name: {key}\n{value}"
            documents.append(Document(page_content=content, metadata={'product_type': product_type, 'product_name': key}))
    return documents

# text chunking
def split_documents(documents, chunk_size=1500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# text embedding
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True},
    )

# FAISS vector db 생성
def create_vector_database(split_documents, embeddings):
    return FAISS.from_documents(documents=split_documents, embedding=embeddings)

# story generation chain
def prepare_story_generation_chain():
    story_examples = [{

        "question":"Create a 4-panel comic strip storyline for a advertisement for Smart Savings Adventure with KB Star Quiz King Savings. Consider the following points: 1. Clear Storyline: A concise and clear structure with a beginning, development, climax, and conclusion. 2. Engaging Characters: Eye-catching design and expressive emotions. 3. Strong Message: Highlight the card benefits (cashback, discounts, points accrual, etc.). 4. Humor and Creativity: Incorporate natural humor and an original, unexpected development. Make each panel consist of a single scene and dialogue. Keep dialouges to 60 characters or less. Describe each scene in detail.",
        "answer":"""
        **cut1_scene**: 
        A lively classroom setting filled with diverse students, aged 14 and older. The teacher, holding a brightly colored poster that says “KB Star Quiz King Savings,” is standing at the front. The students are attentive and eager.
        **cut1_dialogue**: 
        Teacher: \'Who wants to learn how to save money while having fun with quizzes and earning interest up to 10%?\'
        
        **cut2_scene**: 
        A close-up of a student raising their hand enthusiastically. The background shows the blackboard with a chart explaining the savings plan: \'Monthly Savings: 1,000 won to 200,000 won, Interest Rate: 2% to 10%, Duration: 100 days.\'
        **cut2_dialogue**: 
        Student: \'Me! How does it work?\'
        
        **cut3_scene**: 
        The teacher explains while pointing to the blackboard. The blackboard now shows images of coins growing into a stack with a clock indicating the 100-day period.
        **cut3_dialogue**: 
        Teacher: \'It’s simple! Save a small amount every month, solve quizzes to boost your financial knowledge, and watch your savings grow with a high-interest rate.\'
        
        **cut4_scene**: 
        A group of students gathered around a tablet, smiling as they solve a quiz together. On the screen, there’s a congratulatory message: \'You’ve earned 10% interest!” In the background, there’s a banner that reads “KB Star Quiz King Savings - Smart Savings for the Future.
        **cut4_dialogue**: 
        Student: \'Wow, this is awesome! We’re learning and saving at the same time!\'
        """
        
    },
    {
        "question":"Create a 4-panel comic strip storyline for a card advertisement for HanaTour KB Kookmin Card. Consider the following points: 1. Clear Storyline: A concise and clear structure with a beginning, development, climax, and conclusion. 2. Engaging Characters: Eye-catching design and expressive emotions. 3. Strong Message: Highlight the card benefits (cashback, discounts, points accrual, etc.). 4. Humor and Creativity: Incorporate natural humor and an original, unexpected development. Make each panel consist of a single scene and dialogue. Keep dialouges to 60 characters or less. Describe each scene in detail.",
        "answer":"""
        **cut1_scene**: 
        A busy office worker, Sarah, is at her desk, looking stressed with a pile of papers and a computer screen filled with spreadsheets. Her coworker, Mike, peeks over the cubicle wall, looking excited.
        **cut1_dialogue**: '
        Sarah: \'I wish! I'm so overwhelmed with work right now.\'
        
        **cut2_scene**: 
        Sarah at her computer, a thought bubble showing her dream vacation - a tropical beach, clear blue water, and a sun umbrella. She then notices a popup ad for the HanaTour KB Kookmin Card on her screen.
        **cut2_dialogue**: 
        Sarah: \'Hmm, what's this? Special points and discounts with HanaTour KB Kookmin Card?\'
        
        **cut3_scene**: 
        Sarah is now at a HanaTour partner store, holding her new HanaTour KB Kookmin Card, looking thrilled as the cashier hands her a shopping bag with travel gear.
        **cut3_dialogue**: 
        Sarah: \'This is perfect! I can't wait to use my points for my dream vacation.\'
        
        **cut4_scene**: 
        Sarah and Mike at the airport, ready to board their flight. Sarah holds up her card with a big smile. The background shows various benefits of the card, like movie tickets, currency exchange, and interest-free installments.
        **cut4_dialogue**: 
        Sarah: \'Thanks to my HanaTour KB Kookmin Card, I'm finally off on my dream vacation!\'
        """
    }
]

    system_instruction = "You are a comic creator specializing in illustrating benefits of various KB products. Describe every scene in detail."

    example_prompt_story = PromptTemplate(
        input_variables=['question', 'answer'],
        template='question:{question}\nanswer:\n\t{answer}'
    )

    fewshot_prompt_story_gen = FewShotPromptTemplate(
        examples=story_examples,
        example_prompt=example_prompt_story,
        prefix=system_instruction,
        suffix='question:{question}\ncontext:\n{context}\nanswer:\n\t',
        input_variables=['question', 'context']
    )

    llm = ChatOllama(model='gemma2', temperature=1)

    chain = (
        fewshot_prompt_story_gen
        | llm
        | StrOutputParser()
    )

    return chain

# 사용자에게 보여줄 번역 chain
def prepare_translation_chains():
    translation_story_examples = [{
        "question":"""
        **cut1_scene**: 
        A lively classroom setting filled with diverse students, aged 14 and older. The teacher, holding a brightly colored poster that says “KB Star Quiz King Savings,” is standing at the front. The students are attentive and eager.
        **cut1_dialogue**: 
        Teacher: \'Who wants to learn how to save money while having fun with quizzes and earning interest up to 10%?\'
        
        **cut2_scene**: 
        A close-up of a student raising their hand enthusiastically. The background shows the blackboard with a chart explaining the savings plan: \'Monthly Savings: 1,000 won to 200,000 won, Interest Rate: 2% to 10%, Duration: 100 days.\'
        **cut2_dialogue**: 
        Student: \'Me! How does it work?\'
        
        **cut3_scene**: 
        The teacher explains while pointing to the blackboard. The blackboard now shows images of coins growing into a stack with a clock indicating the 100-day period.
        **cut3_dialogue**: 
        Teacher: \'It’s simple! Save a small amount every month, solve quizzes to boost your financial knowledge, and watch your savings grow with a high-interest rate.\'
        
        **cut4_scene**: 
        A group of students gathered around a tablet, smiling as they solve a quiz together. On the screen, there’s a congratulatory message: \'You’ve earned 10% interest!” In the background, there’s a banner that reads “KB Star Quiz King Savings - Smart Savings for the Future.
        **cut4_dialogue**: 
        Student: \'Wow, this is awesome! We’re learning and saving at the same time!\'
        """,
        "answer":"""
        **컷1_장면**:
        다양한 학생들로 가득 찬 생동감 넘치는 교실. 선생님이 "KB 스타 퀴즈킹 저축"이라는 밝은 색상의 포스터를 들고 앞에 서 있다. 학생들은 주의 깊고 열정적이다.
        **컷1_대사**:
        선생님: '퀴즈를 풀면서 저축하고 이자 최대 10%를 얻는 방법을 배우고 싶은 사람?'

        **컷2_장면**:
        열정적으로 손을 든 학생의 클로즈업. 배경에는 월 저축액, 이자율, 기간을 설명하는 차트가 있는 칠판이 보인다: '월 저축액: 1,000원 ~ 200,000원, 이자율: 2% ~ 10%, 기간: 100일.'
        **컷2_대사**:
        학생: '저요! 어떻게 하는 거죠?'

        **컷3_장면**:
        선생님이 칠판을 가리키며 설명한다. 칠판에는 동전이 쌓이는 모습과 100일 기간을 나타내는 시계가 그려져 있다.
        **컷3_대사**:
        선생님: '간단해요! 매달 조금씩 저축하고 퀴즈를 풀어 금융 지식을 높이면서 고이자율로 저축을 늘리는 거예요.'

        **컷4_장면**:
        태블릿을 둘러싼 학생들이 퀴즈를 풀며 웃고 있다. 화면에는 '축하합니다! 10% 이자를 획득했습니다!'라는 메시지가 표시되어 있다. 배경에는 'KB 스타 퀴즈킹 저축 - 미래를 위한 스마트 저축'이라는 배너가 보인다.
        **컷4_대사**:
        학생: '와, 이거 정말 멋지다! 배우면서 저축할 수 있다니!
        """}]


    translation_requirements_examples = [{
        "question":"2번 패널의 대사를 없애줘",
        "answer":"Remove the dialouge in panel 2"
        }]

    translate_llm = ChatOllama(model='gemma2', temperature=0.25, top_k=1, top_p=0.5, mirostat_tau=1, seed=100)

    system_instruction = "You are an expert in English to Korean translation. You need to translate as naturally as possible."

    example_prompt_translate_story = PromptTemplate(
        input_variables=['question', 'answer'],
        template='question:\n{question}\nanswer\n{answer}'
    )

    fewshot_prompt_translation_story = FewShotPromptTemplate(
        examples=translation_story_examples,
        example_prompt=example_prompt_translate_story,
        prefix=system_instruction,
        suffix='question: \n{question}\nanswer:\n',
        input_variables=['question']
    )

    translate_story_chain = (
        fewshot_prompt_translation_story
        | translate_llm
        | StrOutputParser()
    )

    example_prompt_translate_requirements = PromptTemplate(
        input_variables=['question', 'answer'],
        template='question:{question}\nanswer:{answer}'
    )

    fewshot_prompt_translation_requirements = FewShotPromptTemplate(
        examples=translation_requirements_examples,
        example_prompt=example_prompt_translate_requirements,
        prefix=system_instruction,
        suffix='question:{question}\nanswer:',
        input_variables=['question']
    )

    translate_requirements_chain = (
        fewshot_prompt_translation_requirements
        | translate_llm
        | StrOutputParser()
    )

    return translate_story_chain, translate_requirements_chain

# translation 함수
def translate_to_kor(translate_story_chain, question):
    translate_query = f"""Translate the following English story into Korean accurately and completely, without leaving anything out.:
    {question}"""
    response = translate_story_chain.invoke({"question": translate_query})
    return response

def translate_to_eng(translate_requirements_chain, question):
    translate_query = f"""Translate the following Korean question into English accurately and completely, without leaving anything out.: 
    {question}"""
    response = translate_requirements_chain.invoke({"question": translate_query})
    return response

# story generation chain의 query 생성 함수
def create_question(card_name):
    question = f"Create a 4-panel comic strip storyline for a card advertisement for {card_name}. Consider the following points: 1. Clear Storyline: A concise and clear structure with a beginning, development, climax, and conclusion. 2. Engaging Characters: Eye-catching design and expressive emotions. 3. Strong Message: Highlight the card benefits (cashback, discounts, points accrual, etc.). 4. Humor and Creativity: Incorporate natural humor and an original, unexpected development. Make each panel consist of a single scene and dialogue. Keep dialouges to 60 characters or less. Describe each scene in detail."
    return question

def prepare_story_regeneration():
    re_gen_story_examples = [
    {
        "question":"""Don't answer anything else, just make the changes I request in the storyboard and return the entire changed storyboard. Keep dialouges to 60 characters or less. Replace the dialouge in panel 4 from this storyboard:
        **cut1_scene**: 
        A lively classroom setting filled with diverse students, aged 14 and older. The teacher, holding a brightly colored poster that says “KB Star Quiz King Savings,” is standing at the front. The students are attentive and eager.
        **cut1_dialogue**: 
        Teacher: \'Who wants to learn how to save money while having fun with quizzes and earning interest up to 10%?\'
        
        **cut2_scene**: 
        A close-up of a student raising their hand enthusiastically. The background shows the blackboard with a chart explaining the savings plan: \'Monthly Savings: 1,000 won to 200,000 won, Interest Rate: 2% to 10%, Duration: 100 days.\'
        **cut2_dialogue**: 
        Student: \'Me! How does it work?\'
        
        **cut3_scene**: 
        The teacher explains while pointing to the blackboard. The blackboard now shows images of coins growing into a stack with a clock indicating the 100-day period.
        **cut3_dialogue**: 
        Teacher: \'It’s simple! Save a small amount every month, solve quizzes to boost your financial knowledge, and watch your savings grow with a high-interest rate.\'
        
        **cut4_scene**: 
        A group of students gathered around a tablet, smiling as they solve a quiz together. On the screen, there’s a congratulatory message: \'You’ve earned 10% interest!” In the background, there’s a banner that reads “KB Star Quiz King Savings - Smart Savings for the Future.
        **cut4_dialogue**: 
        Student: \'Wow, this is awesome! We’re learning and saving at the same time!\'
        """,
        "answer":"""
        **cut1_scene**: 
        A lively classroom setting filled with diverse students, aged 14 and older. The teacher, holding a brightly colored poster that says “KB Star Quiz King Savings,” is standing at the front. The students are attentive and eager.
        **cut1_dialogue**: 
        Teacher: \'Who wants to learn how to save money while having fun with quizzes and earning interest up to 10%?\'
        
        **cut2_scene**: 
        A close-up of a student raising their hand enthusiastically. The background shows the blackboard with a chart explaining the savings plan: \'Monthly Savings: 1,000 won to 200,000 won, Interest Rate: 2% to 10%, Duration: 100 days.\'
        **cut2_dialogue**: 
        Student: \'Me! How does it work?\'
        
        **cut3_scene**: 
        The teacher explains while pointing to the blackboard. The blackboard now shows images of coins growing into a stack with a clock indicating the 100-day period.
        **cut3_dialogue**: 
        Teacher: \'It’s simple! Save a small amount every month, solve quizzes to boost your financial knowledge, and watch your savings grow with a high-interest rate.\'
        
        **cut4_scene**: 
        A group of students gathered around a tablet, smiling as they solve a quiz together. On the screen, there’s a congratulatory message: \'You’ve earned 10% interest!” In the background, there’s a banner that reads “KB Star Quiz King Savings - Smart Savings for the Future.
        **cut4_dialogue**: 
        Student : "This is amazing! We can earn so much interest just by solving quizzes!"
        """
    }]
    system_instruction= "You are a comic creator specializing in illustrating benefits of various KB products."


    example_re_generation = PromptTemplate(
        input_variables=['question','answer'],
        template='question:{question}\nanswer\n{answer}'
    )


    fewshot_prompt_re_generation = FewShotPromptTemplate(
        examples=re_gen_story_examples,
        example_prompt=example_re_generation,
        prefix=system_instruction, 
        suffix='question:{question}\ncontext:{context}\nanswer:\n',
        input_variables=['question', 'context']
    )
    
    llm = ChatOllama(model='gemma2', temperature=0.5)

    re_generation_chain = (
        fewshot_prompt_re_generation
        |llm
        |StrOutputParser()
    )
    return re_generation_chain

base_dir= '/home/alpaco/cyw/data_files/translated_json'

context = ''
story_chain = ''
translate_story_chain = ''
story_regeneration_chain = ''

def main_fn(cardName):
    global context
    global story_chain
    global translate_story_chain
    global story_regeneration_chain
    global translate_requirements_chain
    documents = load_documents(base_dir)
    split_docs = split_documents(documents)
    embeddings = create_embeddings()
    vector_database = create_vector_database(split_docs, embeddings)

    retriever = vector_database.as_retriever(search_kwargs={'k': 1})
    story_chain = prepare_story_generation_chain()
    story_regeneration_chain= prepare_story_regeneration()
    translate_story_chain, translate_requirements_chain = prepare_translation_chains()


    card_name = cardName
    previous_questions_answers = []

    question = create_question(card_name)
    context = retriever.invoke(card_name)
    response_story = story_chain.invoke({"question": question, "context": context})
    previous_questions_answers.append({
        "question": question, 
        "answer": response_story, 
        "ko_question": translate_to_kor(translate_story_chain, question), 
        "ko_answer": translate_to_kor(translate_story_chain, response_story)
    })
    return previous_questions_answers

def regen(ko_question, previous_questions_answers):
    question= translate_to_eng(translate_requirements_chain, ko_question)
    previous_answer= previous_questions_answers[-1]['answer']
    question= f"Don't answer anything else, just make the changes I request in the storyboard and return the entire changed storyboard. {question} from this storyboard: {previous_answer}"


    answer = story_regeneration_chain.invoke({"question": question, "context": context})
    ko_answer= translate_to_kor(translate_story_chain, answer)
    previous_questions_answers.append({"question": question, "answer": answer,"ko_question":ko_question,"ko_answer":ko_answer})

    return previous_questions_answers