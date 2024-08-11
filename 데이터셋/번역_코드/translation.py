from langchain_community.document_loaders import json_loader
import json
from pathlib import Path
import json
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm
import re


json_filepath= 'data_files/crawling_data/check_card.json'

data= json.loads(Path(json_filepath).read_text())

# json 파일 구조 변경
saving_dict={}
for key, item in data.items():
    
    des_str=''
    for des, description in item.items():
        des_str+=des+':'+description+' '
        des_str+=' '
        des_str= des_str.replace('\n', ' ').replace('card_describe:','')
    saving_dict[key.replace('\n',' ')]= des_str


#example prompts load

prompt_basepath= 'data_files/prompts/'
example_filename= prompt_basepath+ json_filepath.split('/')[-1].split('.')[0]+'_prompt.txt'


with open(example_filename, 'r', encoding='utf-8') as file:
    loaded_data = json.load(file)


examples= loaded_data


#fewshow prompt 

example_prompt = PromptTemplate(
    input_variables=["text", "translation"],
    template='text:{text}\ntranslation:{translation}'
)

# print(example_prompt.format(**examples[0])) # 템플릿 format

system_instruction = "You are a very good translator between Korean and English. You translate the text perfectly into English. I don't need a summary, just translate it exactly as it is in the example prompt."

#prefix=system_instruction,  # 시스템 인스트럭션 추가 가능
prompt = FewShotPromptTemplate(
    examples=examples,
    prefix=system_instruction,
    example_prompt=example_prompt,
    suffix='text:{text} translation:',
    input_variables=['text']
)

# 고유 명사
with open('proper_nouns.json', 'r', encoding='utf-8') as f:
    proper_nouns_dict = json.load(f)

# 모델 chain
llm = ChatOllama(model='gemma2')
chain = prompt | llm | StrOutputParser()




# translate
def translate_data(data, proper_nouns_dict):
    translated_text= translate_text(data ,proper_nouns_dict)
    
    return translated_text


# chunk 별로 split
def split_text_into_chunks(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) + len(word) + 1 > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# 문장별로 data split
def split_text_into_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences


# data를 문장 별로 변환 뒤 translate
def translate_text(text, proper_nouns_dict, max_tokens=200):
    # 고유명사를 영어로 전환
    for noun, translation in proper_nouns_dict.items():
        text = text.replace(noun, translation)

    # 텍스트를 청크로 분할
    chunks = split_text_into_sentences(text)
    translated_chunks = []

    for chunk in chunks:
        # 번역 수행
        translated_chunk = chain.invoke({"text": chunk})
        translated_chunks.append(translated_chunk.split('\n\n\n')[0])

    # 번역된 청크들을 결합
    translated_text = " ".join(translated_chunks)

    return translated_text


# 번역된 결과물이 dictionary로 return
translated_data={}
for item_key in tqdm(saving_dict.keys(), desc="Translating Items"):
    item_data = saving_dict[item_key]
    translated_item = translate_data(item_data, proper_nouns_dict)
    translated_data[proper_nouns_dict[item_key]] = translated_item

for key, value in translated_data.items():
    if 'Please provide the' in value:
        translated_data[key]= value.split('Please provide the')[0]
    
    if 'Please provide me' in value:
        translate_data[key]= value.split('Please provide me')[0]



