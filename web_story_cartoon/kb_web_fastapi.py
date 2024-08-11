from enum import Enum
from fastapi import FastAPI, Form,  Request, Query
from typing_extensions import Annotated
from fastapi.templating import Jinja2Templates
import uvicorn
#from flask import Flask, request, jsonify, redirect, url_for
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from fastapi.staticfiles import StaticFiles
import json
import subprocess
import os
from script.story_generation import main_fn, regen
import time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

templates = Jinja2Templates(directory="templates")



@app.get("/")   #localhost:8000/login 
def test_select(request:Request):
    return templates.TemplateResponse("osh_one.html",{'request':request})

gen_story_list = []
gen_dialogue_list= []

kor_gen_story_list = []
file_list = []
response_story = []
cardN = ''


# 스토리 콘티 생성 함수
def llm_gen_fn(cardN):
    while True:
        try:
            result = subprocess.run("rm /home/alpaco/web_story_cartoon/static/ComfyUI_*.png", shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")

        try:
            result = subprocess.run("rm /home/alpaco/web_story_cartoon/comfyui/ComfyUI/output/ComfyUI_*.png", shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")

        global gen_story_list
        global kor_gen_story_list
        global response_story
        global gen_dialogue_list
        global previous_question_answers


        
        previous_question_answers = main_fn(cardN) 
        result, response_story= previous_question_answers[0]['answer'], previous_question_answers[0]['ko_answer']
        print("===============================")
        print(cardN,result,response_story)
        print("===============================")

        
        gen_story_list = []
        kor_gen_story_list = []
        gen_dialogue_list = []

        for i in result.split("**:"):
            if 'dialogue' in i:
                gen_story_list.append(i.split('**')[0].strip().replace('*',''))
            elif 'scene' in i:
                if 'scene' in i.split('**')[0] or '##' in i.split('**')[0]:
                    pass
                else:
                    gen_dialogue_list.append(i.split('**')[0].strip().replace('*',''))
            else:
                gen_dialogue_list.append(i.strip())
            

        print(gen_story_list)
        
        gen_dialogue_list = []
        for i in response_story.split("**:"):
            if '대사' in i:
                kor_gen_story_list.append(i.split('**')[0].strip().replace('*',''))
            elif '장면' in i:
                if '장면' in i.split('**')[0] or '##' in i.split('**')[0]:
                    pass
                else:
                    gen_dialogue_list.append(i.split('**')[0].strip().replace('*',''))
            else:
                if 'Let me' not in i or '##' not in i or '**' not in i:
                    gen_dialogue_list.append(i.strip())
         
        

            
        print("대사 리스트",gen_dialogue_list)


         
        

        print(kor_gen_story_list)
        print(gen_dialogue_list)

        if len(gen_story_list) < 4 or len(kor_gen_story_list) <  4 or len(gen_dialogue_list) < 4 or len(gen_dialogue_list[0]) == 0 or len(gen_dialogue_list[3])==0:
            continue

        try:
            result = subprocess.run(
                "rm -r /home/alpaco/web_story_cartoon/comfyui/ComfyUI/output/*",
                shell=True,  # Enables shell features, such as wildcard expansion
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")

        try:
            result = subprocess.run(
                "rm -r /home/alpaco/web_story_cartoon/comfyui/ComfyUI/sortlist.txt",
                shell=True,  # Enables shell features, such as wildcard expansion
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")
        


        for i in range(4):
            with open("/home/alpaco/web_story_cartoon/comfyui/ComfyUI/prompt.txt","w") as f:
                f.write(gen_story_list[i])

            # 실행할 Python 스크립트 파일 경로
            script_path = "comfy_run.py"

            # Python 명령어 실행
            try:
                result = subprocess.run(
                    ["/home/alpaco/miniconda3/envs/comfy_main/bin/python", script_path],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd="/home/alpaco/web_story_cartoon/comfyui/ComfyUI",
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"}  # 2번째 GPU 사용 설정
            # 이동할 디렉토리 경로 지정
                )
            except subprocess.CalledProcessError as e:
                print(f"Error Output:\n{e.stderr}")
            time.sleep(1)

        global file_list
        with open("/home/alpaco/web_story_cartoon/comfyui/ComfyUI/sortlist.txt","r") as f:
            file_list = f.readlines()

        try:
            result = subprocess.run(
                "cp -r /home/alpaco/web_story_cartoon/comfyui/ComfyUI/output/* /home/alpaco/web_story_cartoon/static/",
                shell=True,  # Enables shell features, such as wildcard expansion
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")
    
        for i in range(len(file_list)):
            if i != 3:
                try:
                    result = subprocess.run(
                            [
                                "python",
                                "script/bubble_scene123.py",
                                "--cartoon_image", f"/home/alpaco/web_story_cartoon/static/{file_list[i].strip()}",
                                "--output_path", f"/home/alpaco/web_story_cartoon/static/{file_list[i].strip()}",
                                "--text", gen_dialogue_list[i].strip()
                            ],
                            check=True,
                            text=True
                        )
                except subprocess.CalledProcessError as e:
                    print(f"Error Output:\n{e.stderr}")
            else:
                try:
                    result = subprocess.run(
                            [
                                "python",
                                "script/bubble_card_scene4.py",
                                "--cartoon_image", f"/home/alpaco/web_story_cartoon/static/{file_list[i].strip()}",
                                "--output_path", f"/home/alpaco/web_story_cartoon/static/{file_list[i].strip()}",
                                "--text", gen_dialogue_list[i].strip()
                            ],
                            check=True,
                            text=True
                        )
                except subprocess.CalledProcessError as e:
                    print(f"Error Output:\n{e.stderr}")
        #time.sleep(2)

        try:
            result = subprocess.run(
                "rm -r /home/alpaco/web_story_cartoon/__pycache__",
                shell=True,  # Enables shell features, such as wildcard expansion
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")

        return kor_gen_story_list,response_story,file_list

# 특정 패널에 대해서 콘티를 재생성
def REGEN_llm_gen_fn(question):
    while True:
        try:
            result = subprocess.run("rm /home/alpaco/web_story_cartoon/static/ComfyUI_*.png", shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")

        try:
            result = subprocess.run("rm /home/alpaco/web_story_cartoon/comfyui/ComfyUI/output/ComfyUI_*.png", shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")

        global gen_story_list
        global kor_gen_story_list
        global response_story
        global gen_dialogue_list
        global previous_question_answers
  

        previous_question_answers = regen(question, previous_question_answers)
        result, response_story=previous_question_answers[-1]['answer'],previous_question_answers[-1]['ko_answer']
        #print(result)
    
        print("===============================")
        print(result,response_story)
        print("===============================")

        gen_story_list = []
        kor_gen_story_list = []
        gen_dialogue_list = []

        for i in result.split("**:"):
            if 'dialogue' in i:
                gen_story_list.append(i.split('**')[0].strip().replace('*',''))
            elif 'scene' in i:
                if 'scene' in i.split('**')[0] or '##' in i.split('**')[0]:
                    pass
                else:
                    gen_dialogue_list.append(i.split('**')[0].strip().replace('*',''))
            else:
                gen_dialogue_list.append(i.strip())
            

        print(gen_story_list)
        
        gen_dialogue_list = []
        for i in response_story.split("**:"):
            if '대사' in i:
                kor_gen_story_list.append(i.split('**')[0].strip().replace('*',''))
            elif '장면' in i:
                if '장면' in i.split('**')[0] or '##' in i.split('**')[0]:
                    pass
                else:
                    gen_dialogue_list.append(i.split('**')[0].strip().replace('*',''))
            else:
                if 'Let me' not in i or '##' not in i or '**' not in i:
                    gen_dialogue_list.append(i.strip())
         
        

            
        print("대사 리스트",gen_dialogue_list)


         
        

        print(kor_gen_story_list)
        print(gen_dialogue_list)

        if len(gen_story_list) < 4 or len(kor_gen_story_list) <  4 or len(gen_dialogue_list) < 4 or len(gen_dialogue_list[0]) == 0 or len(gen_story_list[3]) == 0:
            continue

        try:
            result = subprocess.run(
                "rm -r /home/alpaco/web_story_cartoon/comfyui/ComfyUI/output/*",
                shell=True,  # Enables shell features, such as wildcard expansion
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")

        try:
            result = subprocess.run(
                "rm -r /home/alpaco/web_story_cartoon/comfyui/ComfyUI/sortlist.txt",
                shell=True,  # Enables shell features, such as wildcard expansion
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")

        
        for i in range(4):
            with open("/home/alpaco/web_story_cartoon/comfyui/ComfyUI/prompt.txt","w") as f:
                f.write(gen_story_list[i])

            # 실행할 Python 스크립트 파일 경로
            script_path = "comfy_run.py"

            # Python 명령어 실행
            try:
                result = subprocess.run(
                    ["/home/alpaco/miniconda3/envs/comfy_main/bin/python", script_path],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd="/home/alpaco/web_story_cartoon/comfyui/ComfyUI",
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"}  # 2번째 GPU 사용 설정
            # 이동할 디렉토리 경로 지정
                )
            except subprocess.CalledProcessError as e:
                print(f"Error Output:\n{e.stderr}")

        global file_list
        with open("/home/alpaco/web_story_cartoon/comfyui/ComfyUI/sortlist.txt","r") as f:
            file_list = f.readlines()

        
        try:
            result = subprocess.run(
                "cp -r /home/alpaco/web_story_cartoon/comfyui/ComfyUI/output/* /home/alpaco/web_story_cartoon/static/",
                shell=True,  # Enables shell features, such as wildcard expansion
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")
    

        for i in range(len(file_list)):
            if i != 3:
                try:
                    #result = subprocess.run(f"python script/bubble_scene123.py --cartoon_image /home/alpaco/web_story_cartoon/static/{file_list[i].strip()} --output_path /home/alpaco/web_story_cartoon/static/{file_list[i].strip()} --text \'{gen_dialogue_list[i].strip()}\'", shell=True)
                    result = subprocess.run(
                            [
                                "python",
                                "script/bubble_scene123.py",
                                "--cartoon_image", f"/home/alpaco/web_story_cartoon/static/{file_list[i].strip()}",
                                "--output_path", f"/home/alpaco/web_story_cartoon/static/{file_list[i].strip()}",
                                "--text", gen_dialogue_list[i].strip()
                            ],
                            check=True,
                            text=True
                        )
                except subprocess.CalledProcessError as e:
                    print(f"Error Output:\n{e.stderr}")
            else:
                try:
                    #result = subprocess.run(f"python script/bubble_card_scene4.py --cartoon_image /home/alpaco/web_story_cartoon/static/{file_list[i].strip()} --output_path /home/alpaco/web_story_cartoon/static/{file_list[i].strip()} --text \'{gen_dialogue_list[i].strip()}\'", shell=True)
                    result = subprocess.run(
                            [
                                "python",
                                "script/bubble_card_scene4.py",
                                "--cartoon_image", f"/home/alpaco/web_story_cartoon/static/{file_list[i].strip()}",
                                "--output_path", f"/home/alpaco/web_story_cartoon/static/{file_list[i].strip()}",
                                "--text", gen_dialogue_list[i].strip()
                            ],
                            check=True,
                            text=True
                        )
                except subprocess.CalledProcessError as e:
                    print(f"Error Output:\n{e.stderr}")


        try:
            result = subprocess.run(
                "rm -r /home/alpaco/web_story_cartoon/__pycache__",
                shell=True,  # Enables shell features, such as wildcard expansion
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")

        return kor_gen_story_list,response_story,file_list


# 특정 이미지를 선택해 특정 이미지만 재생성
@app.post("/oneregenImage")
async def oneregen_image(request:Request,selectedImg: int = Form(...), positivePrompt: str = Form(...), negativePrompt: str = Form(...)):
    # 선택한 이미지 번호, 긍정 프롬프트, 부정 프롬프트를 받음
    selectedImg = selectedImg-1
    with open("/home/alpaco/web_story_cartoon/comfyui/ComfyUI/prompt.txt","w") as f:
        f.write(positivePrompt)
        f.write(negativePrompt)

    # 1개 생성 
    script_path = "comfy_run.py"

    # Python 명령어 실행
    try:
        result = subprocess.run(
            ["/home/alpaco/miniconda3/envs/comfy_main/bin/python", script_path],
            capture_output=True,
            text=True,
            check=True,
            cwd="/home/alpaco/web_story_cartoon/comfyui/ComfyUI",
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"}  # 2번째 GPU 사용 설정
        )
    except subprocess.CalledProcessError as e:
        print(f"Error Output:\n{e.stderr}")
            
    with open("/home/alpaco/web_story_cartoon/comfyui/ComfyUI/sortlist.txt","r") as f:
        new_file_list = f.readlines()
    
    newfile = set(new_file_list) - set(file_list) 
    newfile = list(newfile)[0].replace("\n","")
    print("다른거:",set(new_file_list))
    print("다른거:",set(file_list))
    print("다른거:",newfile)

    try:
        result = subprocess.run(
            f"rm -r /home/alpaco/web_story_cartoon/comfyui/ComfyUI/output/{file_list[selectedImg]}",
            shell=True,  # Enables shell features, such as wildcard expansion
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error Output:\n{e.stderr}")

    try:
        result = subprocess.run(
            f"rm -r /home/alpaco/web_story_cartoon/static/{file_list[selectedImg]}",
            shell=True,  # Enables shell features, such as wildcard expansion
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error Output:\n{e.stderr}")


    file_list[selectedImg] = newfile

    with open("/home/alpaco/web_story_cartoon/comfyui/ComfyUI/sortlist.txt","w") as f:
        for i in file_list:
            f.write(i.replace("\n","")+"\n")

    try:
        result = subprocess.run(
            f"cp -r /home/alpaco/web_story_cartoon/comfyui/ComfyUI/output/{newfile} /home/alpaco/web_story_cartoon/static/{newfile}",
            shell=True,  # Enables shell features, such as wildcard expansion
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error Output:\n{e.stderr}")
 

    if selectedImg == 3 :
        try:
            #result = subprocess.run(f"python script/bubble_scene123.py --cartoon_image /home/alpaco/web_story_cartoon/static/{newfile.strip()} --output_path /home/alpaco/web_story_cartoon/static/{newfile.strip()} --text \'{gen_dialogue_list[selectedImg].strip()}\'", shell=True)
            result = subprocess.run(
                [
                    "python",
                    "script/bubble_card_scene4.py",
                    "--cartoon_image", f"/home/alpaco/web_story_cartoon/static/{newfile.strip()}",
                    "--output_path", f"/home/alpaco/web_story_cartoon/static/{newfile.strip()}",
                    "--text", gen_dialogue_list[selectedImg].strip()
                ],
                check=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")
    else:
        try:
            #result = subprocess.run(f"python script/bubble_card_scene4.py --cartoon_image /home/alpaco/web_story_cartoon/static/{newfile.strip()} --output_path /home/alpaco/web_story_cartoon/static/{newfile.strip()} --text \'{gen_dialogue_list[selectedImg].strip()}\'", shell=True)
            result = subprocess.run(
                [
                    "python",
                    "script/bubble_scene123.py",
                    "--cartoon_image", f"/home/alpaco/web_story_cartoon/static/{newfile.strip()}",
                    "--output_path", f"/home/alpaco/web_story_cartoon/static/{newfile.strip()}",
                    "--text", gen_dialogue_list[selectedImg].strip()
                ],
                check=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error Output:\n{e.stderr}")




    #kor_gen_story_list,response_story,file_list = llm_gen_fn(cardN)
    time.sleep(3)
    # 임시로 성공 메시지를 반환
    return templates.TemplateResponse("osh_second.html", {"request": request, 
                                                           "message": f"Card '{cardN}' processed successfully.",
                                                           "response_story": response_story,
                                                           "gen_story1":kor_gen_story_list[0],
                                                           "gen_story2":kor_gen_story_list[1],
                                                           "gen_story3":kor_gen_story_list[2],
                                                           "gen_story4":kor_gen_story_list[3],
                                                           "img1":file_list[0].split("/")[-1],
                                                           "img2":file_list[1].split("/")[-1],
                                                           "img3":file_list[2].split("/")[-1],
                                                           "img4":file_list[3].split("/")[-1],})




# 특정 패널에 대한 콘티를 재생성 
@app.post("/onecheckcard")
async def handle_form(request: Request,
    modifyInput: str = Form(None)
):
    kor_gen_story_list,response_story,file_list = REGEN_llm_gen_fn(modifyInput)
    
    return templates.TemplateResponse("osh_second.html", {"request": request, 
                                                           "message": f"Card '{cardN}' processed successfully.",
                                                           "response_story": response_story,
                                                           "gen_story1":kor_gen_story_list[0],
                                                           "gen_story2":kor_gen_story_list[1],
                                                           "gen_story3":kor_gen_story_list[2],
                                                           "gen_story4":kor_gen_story_list[3],
                                                           "img1":file_list[0].split("/")[-1],
                                                           "img2":file_list[1].split("/")[-1],
                                                           "img3":file_list[2].split("/")[-1],
                                                           "img4":file_list[3].split("/")[-1],})



## 스토리 전체 다시 생성하는곳
@app.get("/checkcard")
async def get_check_card(request: Request, cardN: str = Query(None)):
    #print("heelod")
    global card_name_global
    kor_gen_story_list,response_story,file_list = llm_gen_fn(card_name_global)
    time.sleep(3)

    
    # 임시로 성공 메시지를 반환
   
    return templates.TemplateResponse("osh_second.html", {"request": request, 
                                                           "message": f"Card '{cardN}' processed successfully.",
                                                           "response_story": response_story,
                                                           "gen_story1":kor_gen_story_list[0],
                                                           "gen_story2":kor_gen_story_list[1],
                                                           "gen_story3":kor_gen_story_list[2],
                                                           "gen_story4":kor_gen_story_list[3],
                                                           "img1":file_list[0].split("/")[-1],
                                                           "img2":file_list[1].split("/")[-1],
                                                           "img3":file_list[2].split("/")[-1],
                                                           "img4":file_list[3].split("/")[-1],})
    


@app.post("/checkcard")
async def check_card(request: Request, cardN: str = Form(...)):
    # 여기에서 입력된 카드 이름에 대한 로직을 구현할 수 있습니다.
    # 예: 데이터베이스에서 카드 이름을 검증하거나 특정 작업 수행
    print(cardN)
    with open('cardname_to_eng.json', 'r', encoding='utf-8') as f:
        card_name_to_eng = json.load(f)
    cardN= card_name_to_eng[cardN]
    global card_name_global
    card_name_global = cardN

    kor_gen_story_list,response_story,file_list = llm_gen_fn(cardN)
    print(file_list)


    return templates.TemplateResponse("osh_second.html", {"request": request, 
                                                           "message": f"Card '{cardN}' processed successfully.",
                                                           "response_story": response_story,
                                                           "gen_story1":kor_gen_story_list[0],
                                                           "gen_story2":kor_gen_story_list[1],
                                                           "gen_story3":kor_gen_story_list[2],
                                                           "gen_story4":kor_gen_story_list[3],
                                                           "img1":file_list[0].split("/")[-1],
                                                           "img2":file_list[1].split("/")[-1],
                                                           "img3":file_list[2].split("/")[-1],
                                                           "img4":file_list[3].split("/")[-1],})

# 만화 생성 완료 후 DB에 저장                                                
@app.get("/complete")
async def complete(request: Request):

    try:
        result = subprocess.run(f"python cartoon_combine.py \
         --scene1_path ./output_img/result_s1.png --scene2_path ./output_img/result_s2.png --scene3_path ./output_img/result_s3.png --scene4_path ./output_img/result_s4.png --output_path ./output_img/final_cartoon.png", shell=True)
        
        result = subprocess.run(
                [
                    "python",
                    "script/cartoon_combine.py",
                    "--scene1_path", "./static/"+file_list[0].split("/")[-1].strip(),
                    "--scene2_path", "./static/"+file_list[1].split("/")[-1].strip(),
                    "--scene3_path", "./static/"+file_list[2].split("/")[-1].strip(),
                    "--scene4_path", "./static/"+file_list[3].split("/")[-1].strip(),
                    "--output_path", "./static/result_img.png", 
                ],
                check=True,
                text=True
            )
    except subprocess.CalledProcessError as e:
        print(f"Error Output:\n{e.stderr}")

    return templates.TemplateResponse("osh_third.html", {"request": request, 
                                                        "result_img": "result_img.png",
                                                           "img1":file_list[0].split("/")[-1].strip(),
                                                           "img2":file_list[1].split("/")[-1].strip(),
                                                           "img3":file_list[2].split("/")[-1].strip(),
                                                           "img4":file_list[3].split("/")[-1].strip(),})

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port = 8000)


