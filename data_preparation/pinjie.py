import cv2
import os
import numpy as np
from PIL import Image
folder_list = []
# 输入文件夹路径和输出文件夹路径
input_folder1 = "vis_paper_levir/BIT/A"
input_folder2 = "vis_paper_levir/BIT/B"
input_folder3 = "vis_paper_levir/BIT/gt"
input_folder4 = "vis_paper_levir/BIT/pred"
input_folder5 = "vis_paper_levir/ICIF-Net/pred"
input_folder6 = "vis_paper_levir/EGCTNet/pred"
input_folder7 = "vis_paper_levir/ChangeFormer/pred"
input_folder8 = "vis_paper_levir/EGENet/pred"
folder_list.append(input_folder1)
folder_list.append(input_folder2)
folder_list.append(input_folder3)
folder_list.append(input_folder4)
folder_list.append(input_folder5)
folder_list.append(input_folder6)
folder_list.append(input_folder7)
folder_list.append(input_folder8)
gate = 128
image_files = sorted([f for f in os.listdir(input_folder1) if f.endswith('.jpg')])
for image_file in image_files:
    num_images = 8
    image_size = 256  
    margin = 3  # 间隙大小
    big_image_width = num_images * (image_size + margin) - margin
    big_image_height = image_size
    big_image = Image.new('RGB', (big_image_width, big_image_height), (255, 255, 255))
    true_change_image_path = os.path.join(input_folder3, image_file)
    true_change_image = cv2.imread(true_change_image_path, cv2.IMREAD_GRAYSCALE)
    i = 0
    for fold in folder_list:
    # 打开小图片
        small_image_path = os.path.join(fold, image_file)
        small_image = Image.open(small_image_path)
        # 计算小图片在大图中的位置
        left = i * (image_size + margin)
        upper = 0
        right = left + image_size
        lower = upper + image_size

        if i<3:
            big_image.paste(small_image, (left, upper, right, lower))
        # 关闭小图片
        small_image.close()

        # 读取预测图
        if i >= 3:
            pred_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)

            result_image = np.zeros((true_change_image.shape[0], true_change_image.shape[1], 3), dtype=np.uint8)

            result_image[(true_change_image >gate) & (pred_image >gate)] = [255, 255, 255]  # White: True Positive (TP)
            result_image[(true_change_image >gate) & (pred_image == 0)] = [0, 255, 0]        # Green: False Negative (FN)
            result_image[(true_change_image == 0) & (pred_image == 0)] = [0, 0, 0]            # Black: True Negative (TN)
            result_image[(true_change_image == 0) & (pred_image >gate)] = [0, 0, 255]        # Blue: False Positive (FP)
            result_overlay = Image.fromarray(result_image)
            big_image.paste(result_overlay, (left, upper, right, lower))
        i = i+1

# 保存大图
    big_image.save('vis_paper_levir/result/'+image_file)

# 关闭大图
    big_image.close()
