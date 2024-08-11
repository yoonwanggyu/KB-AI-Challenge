# python bubble_scene123.py --cartoon_image ./input_img/0807_s1.png --output_path ./output_img/result_s1.png --text "으아, 청구서들이 쌓여가고 있어! 돈을 절약할 방법이 필요해..."
# python bubble_scene123.py --cartoon_image ./input_img/0807_s2.png --output_path ./output_img/result_s2.png --text "톡톡 카드? 내가 필요로 하는 건 바로 이거 같아!"
# python bubble_scene123.py --cartoon_image ./input_img/0807_s3.png --output_path ./output_img/result_s3.png --text "정말! 톡톡 카드 덕분에 라떼 가격이 절반으로 내려갔어!"

import argparse
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFont, ImageDraw


# 커맨드 라인 인자 설정
def parse_args():
    parser = argparse.ArgumentParser(description='Process some images and add text with a speech bubble.')
    parser.add_argument('--cartoon_image', type=str, required=True, help='Path to the cartoon image file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path where the output image will be saved.')
    parser.add_argument('--text', type=str, required=True, help='Text to be added to the speech bubble.')
    return parser.parse_args()

# 인자 파싱
args = parse_args()


# Inference client 설정
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="16O3Z2XxorCJlaTYJZEc"
)

# 이미지 로드
cartoon_img = Image.open(args.cartoon_image) # 이미지 로드 
bubble_image_path = r"./input_img/bubble.png"  # 실제 말풍선 이미지 경로
bubble_img = Image.open(bubble_image_path) 
output_image_path = args.output_path # 이미지가 저장될 경로

# face detection
custom_configuration = InferenceConfiguration(confidence_threshold=0.2) # confidence threshold 0.2
with CLIENT.use_configuration(custom_configuration):
    result = CLIENT.infer(args.cartoon_image, model_id="only-faces/5")

# 텍스트 설정
text = args.text
if ':' in text:
    text= text.split(':')[1]
font_path = r"./input_img/MaruBuri-Light.ttf"  # 설치된 폰트 경로


# 이미지 크기와 예측 결과
image_width, image_height = cartoon_img.size
predictions = result['predictions']
predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)


# 말풍선을 만화 이미지에 추가하는 함수
def add_speech_bubble(ax, cartoon_img, bubble_img, text, position, font_path):
    bubble_x, bubble_y, bubble_width = position
    bubble_aspect_ratio = bubble_img.width / bubble_img.height
    bubble_height = bubble_width / bubble_aspect_ratio

    bubble_img_resized = bubble_img.resize((int(bubble_width), int(bubble_height)))

    # 리사이즈된 말풍선 이미지의 크기를 기준으로 폰트 크기 조정
    bubble_width_resized, bubble_height_resized = bubble_img_resized.size

    combined_img = cartoon_img.copy()
    combined_img.paste(bubble_img_resized, (int(bubble_x), int(bubble_y)), bubble_img_resized)

    ax.imshow(combined_img)

    # 적절한 비율로 폰트 크기 조정
    ratio = 0.15
    font_size = int((bubble_width_resized / 10) * ratio)  # 기본 폰트 크기를 비율에 맞게 조정

    font = ImageFont.truetype(font_path, font_size)
    lines = split_text(text, font, bubble_width_resized * ratio)  # 여백을 위해 15%를 사용

    # 텍스트 높이 계산
    text_height = sum([font.getsize(line)[1] for line in lines])
    text_height += (len(lines) - 1) * (font.getsize(text)[1] * 0.2)  # 줄간격을 고려한 추가 높이

    # 텍스트 중앙 정렬
    crr_y = bubble_y + (bubble_height_resized - text_height) / 2
    for line in lines: 
        ax.text(bubble_x + bubble_width_resized / 2, crr_y - len(lines)*10 - 10, line, fontsize=font_size, fontproperties={'fname': font_path}, va='top', ha='center', color='black')
        crr_y += font.getsize(line)[1] * 3 # 줄간격을 3배로 설정
        
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

def split_text(text, font, max_width):
    # 텍스트를 줄바꿈하기 위한 함수
    words = text.split()
    lines = []
    current_line = words[0]

    for word in words[1:]:
        test_line = current_line + " " + word
        width = font.getsize(test_line)[0]
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

# 결과 이미지 출력
fig, ax = plt.subplots(1)


prediction = predictions[0]

x_center = prediction['x']
y_center = prediction['y']
width = prediction['width']
height = prediction['height']

x = x_center - width / 2
y = y_center - height / 2

if x_center < image_width / 2 and y_center > image_height / 2:
    right_edge = x + width
    margin = image_width - right_edge

    position = (right_edge, y_center - height / 2, margin)  # 위치 조정
    add_speech_bubble(ax, cartoon_img, bubble_img, text, position, font_path)

elif x_center > image_width / 2 and y_center < image_height / 2:
    left_edge = x
    margin = left_edge

    position = (0, 0, margin)  # 위치 조정
    add_speech_bubble(ax, cartoon_img, bubble_img, text, position, font_path)

elif x_center < image_width / 2 and y_center < image_height / 2:
    right_edge = x + width
    margin = image_width - right_edge

    position = (right_edge, 0, margin)  # 위치 조정
    add_speech_bubble(ax, cartoon_img, bubble_img, text, position, font_path)

elif x_center > image_width / 2 and y_center > image_height / 2:
    left_edge = x
    margin = left_edge

    position = (left_edge - margin, y_center - height, margin)  # 위치 조정
    add_speech_bubble(ax, cartoon_img, bubble_img, text, position, font_path)

