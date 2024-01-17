from PIL import Image
import os

# 输入文件夹路径和输出文件夹路径
input_folder = 'D://文件库//file//下载//ICIF-Net_LEVIR//LEVIR'
output_folders = ['vis_paper_levir/ICIF-Net/A', 'vis_paper_levir/ICIF-Net/B', 'vis_paper_levir/ICIF-Net/gt', 'vis_paper_levir/ICIF-Net/pred']

# 获取输入文件夹中的所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# 遍历每个图片文件
for image_file in image_files:
    # 打开图片
    image_path = os.path.join(input_folder, image_file)
    img = Image.open(image_path)

    # 获取图片大小
    width, height = img.size

    # 计算列数
    num_cols = width // 256

    # 切分小图片
    for row in range(4):
        for col in range(num_cols):
            # 计算小图片的位置
            left = col * 256
            upper = row * 256
            right = left + 256
            lower = upper + 256

            # 切分小图片
            small_img = img.crop((left, upper, right, lower))

            # 保存小图片到相应的输出文件夹
            output_folder = output_folders[row]
            output_path = os.path.join(output_folder, f'{col+1}_{image_file}')
            small_img.save(output_path)

    # 关闭图片
    img.close()
