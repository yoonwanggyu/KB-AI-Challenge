# python cartoon_combine.py --scene1_path ./output_img/result_s1.png --scene2_path ./output_img/result_s2.png --scene3_path ./output_img/result_s3.png --scene4_path ./output_img/result_s4.png --output_path ./output_img/final_cartoon.png

import argparse
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 커맨드 라인 인자 설정
def parse_args():
    parser = argparse.ArgumentParser(description='Process combine 4 cartoons')
    parser.add_argument('--scene1_path', type=str, required=True, help='Path to the cartoon image file.')
    parser.add_argument('--scene2_path', type=str, required=True, help='Path to the cartoon image file.')
    parser.add_argument('--scene3_path', type=str, required=True, help='Path to the cartoon image file.')
    parser.add_argument('--scene4_path', type=str, required=True, help='Path to the cartoon image file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path where the output image will be saved.')
    return parser.parse_args()


# 인자 파싱
args = parse_args()

# 저장할 경로
save_path = args.output_path

# 이미지 로드
s1_img = Image.open(args.scene1_path)
s2_img = Image.open(args.scene2_path)
s3_img = Image.open(args.scene3_path)
s4_img = Image.open(args.scene4_path)

# 각 이미지에 검정색 테두리 추가
border_size = 3
s1_img = ImageOps.expand(s1_img, border=border_size, fill='black')
s2_img = ImageOps.expand(s2_img, border=border_size, fill='black')
s3_img = ImageOps.expand(s3_img, border=border_size, fill='black')
s4_img = ImageOps.expand(s4_img, border=border_size, fill='black')

# 각 이미지의 너비와 높이를 얻기
s1_width, s1_height = s1_img.size
s2_width, s2_height = s2_img.size
s3_width, s3_height = s3_img.size
s4_width, s4_height = s4_img.size

# 최대 너비와 최대 높이 계산
max_width = max(s1_width, s3_width)
max_height = max(s1_height, s2_height)

# 이미지 사이의 간격 설정
spacing = 15

# 새 이미지의 전체 크기 계산 (간격 포함)
total_width = 2 * max_width + spacing
total_height = 2 * max_height + spacing

# 새 이미지를 위한 빈 캔버스 생성 (배경은 흰색)
background_color = (255, 255, 255)
new_img = Image.new('RGB', (total_width, total_height), background_color)

# 각 이미지를 적절한 위치에 붙이기
new_img.paste(s1_img, (0, 0))
new_img.paste(s2_img, (max_width + spacing, 0))
new_img.paste(s3_img, (0, max_height + spacing))
new_img.paste(s4_img, (max_width + spacing, max_height + spacing))

# 결과 이미지 출력 및 저장
fig, ax = plt.subplots(1)
ax.imshow(new_img)

plt.axis('off')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.08,dpi=300)
