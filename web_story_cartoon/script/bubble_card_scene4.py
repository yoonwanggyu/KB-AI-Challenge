# python bubble_card_scene4.py --cartoon_image ./input_img/0807_s4.png --output_path ./output_img/result_s4.png --text "톡톡 카드가 일상생활을 더 저렴하고 즐겁게 만들어줘!"


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
bubble_image_path = r"./input_img/wide_bubble.png"  # 실제 말풍선 이미지 경로
card_image_path = r"./input_img/toktok_card.png" # 실제 카드 이미지 경로
cartoon_img = Image.open(args.cartoon_image).convert("RGBA")
bubble_img = Image.open(bubble_image_path).convert("RGBA")
card_img = Image.open(card_image_path).convert("RGBA")
output_image_path = args.output_path

# face detection
custom_configuration = InferenceConfiguration(confidence_threshold=0.3) # confidence threshold 0.3
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
def add_bubble_card(ax, cartoon_img, bubble_img, text, position, font_path):
    bubble_x, bubble_y, bubble_width = position
    bubble_aspect_ratio = bubble_img.width / bubble_img.height
    bubble_height = bubble_width / bubble_aspect_ratio

    bubble_img_resized = bubble_img.resize((int(bubble_width*0.75) , int(bubble_height*0.75)))

    # 리사이즈된 말풍선 이미지의 크기를 기준으로 폰트 크기 조정
    bubble_width_resized, bubble_height_resized = bubble_img_resized.size

    combined_img = cartoon_img.copy()
    combined_img.paste(bubble_img_resized, (int(bubble_x), int(bubble_y)), bubble_img_resized)
    
    # 카드 이미지 크기 조정 (1/4)
    card_width, card_height = combined_img.size
    new_card_width = card_width // 4
    new_card_height = card_height // 4
    card_img_resized = card_img.resize((new_card_width, new_card_height), Image.ANTIALIAS)

    # 카드 이미지 배치 위치 계산 (오른쪽 하단)
    position = (combined_img.width - new_card_width - 30, combined_img.height - new_card_height - 20)

    # alpha 채널을 마스크로 사용하여 카드 이미지 붙이기
    combined_img.paste(card_img_resized, position, card_img_resized)

    ax.imshow(combined_img)

    # 적절한 비율로 폰트 크기 조정
    ratio = 0.25
    font_size = int((bubble_width_resized / 10) * ratio)  # 기본 폰트 크기를 비율에 맞게 조정

    font = ImageFont.truetype(font_path, font_size)
    lines = split_text(text, font, bubble_width_resized * ratio)  # 여백을 위해 25%를 사용

    # 텍스트 높이 계산
    text_height = sum([font.getsize(line)[1] for line in lines])
    text_height += (len(lines) - 1) * (font.getsize(text)[1] * 0.2)  # 줄간격을 고려한 추가 높이

    # 텍스트 중앙 정렬
    crr_y = bubble_y + (bubble_height_resized - text_height) / 2
    for line in lines:
        ax.text(bubble_x + bubble_width_resized / 2, crr_y- len(lines)*10 - 10, line, fontsize=font_size, fontproperties={'fname': font_path}, va='top', ha='center', color='black')
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
    add_bubble_card(ax, cartoon_img, bubble_img, text, position, font_path)

elif x_center > image_width / 2 and y_center < image_height / 2:
    left_edge = x
    margin = left_edge

    position = (0, 0, margin)  # 위치 조정
    add_bubble_card(ax, cartoon_img, bubble_img, text, position, font_path)

elif x_center < image_width / 2 and y_center < image_height / 2:
    right_edge = x + width
    margin = image_width - right_edge

    position = (right_edge, 0, margin)  # 위치 조정
    add_bubble_card(ax, cartoon_img, bubble_img, text, position, font_path)

elif x_center > image_width / 2 and y_center > image_height / 2:
    left_edge = x
    margin = left_edge

    position = (left_edge - margin, y_center - height, margin)  # 위치 조정
    add_bubble_card(ax, cartoon_img, bubble_img, text, position, font_path)
