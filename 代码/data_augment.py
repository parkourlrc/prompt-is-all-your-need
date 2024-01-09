import os
import cv2
import shutil
# 源文件夹路径
source_folder = '/content/drive/MyDrive/城市工程系统智能化/Soil types train'
# 目标文件夹路径
destination_folder = '/content/drive/MyDrive/城市工程系统智能化/augmented train datas'
# 数据增强参数
rotation_angles = [90, 180, 270]  # 旋转角度
flip_modes = [0, 1, -1]  # 翻转模式：0表示垂直翻转，1表示水平翻转，-1表示水平垂直翻转
brightness_factors = [0.7, 1.3]  # 亮度变化因子
crop_ratios = [0.8, 0.9, 1.1, 1.2]  # 裁剪比例
# 遍历源文件夹中的每个文件夹
for root, _, files in os.walk(destination_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        # 仅处理图像文件
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 读取图像
            image = cv2.imread(file_path)
            # 生成增广后的图像
            augmented_images = []
            # 旋转增广
            for angle in rotation_angles:
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) if angle == 90 else cv2.rotate(image, cv2.ROTATE_180) if angle == 180 else cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                augmented_images.append(rotated_image)
            # 生成增广后的文件名和文件路径，并保存增广后的图像
            for i, augmented_image in enumerate(augmented_images):
                augmented_file_name = f"augmented_{i}_{file_name}"
                augmented_file_path = os.path.join(root, augmented_file_name)
                cv2.imwrite(augmented_file_path, augmented_image)
for root, _, files in os.walk(destination_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image = cv2.imread(file_path)
            augmented_images = []
            # 翻转增广
            for flip_mode in flip_modes:
                flipped_image = cv2.flip(image, flip_mode)
                augmented_images.append(flipped_image)
            for i, augmented_image in enumerate(augmented_images):
                augmented_file_name = f"augmented_{i}_{file_name}"
                augmented_file_path = os.path.join(root, augmented_file_name)
                cv2.imwrite(augmented_file_path, augmented_image)
for root, _, files in os.walk(destination_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image = cv2.imread(file_path)
            augmented_images = []
            # 亮度变化增广
            for factor in brightness_factors:
                brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
                augmented_images.append(brightened_image)
            for i, augmented_image in enumerate(augmented_images):
                augmented_file_name = f"augmented_{i}_{file_name}"
                augmented_file_path = os.path.join(root, augmented_file_name)
                cv2.imwrite(augmented_file_path, augmented_image)
for root, _, files in os.walk(destination_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image = cv2.imread(file_path)
            augmented_images = []
            # 裁剪增广
            height, width, _ = image.shape
            for ratio in crop_ratios:
                if ratio < 1:
                    new_height = int(height * ratio)
                    new_width = int(width * ratio)
                    y = int((height - new_height) / 2)
                    x = int((width - new_width) / 2)
                    cropped_image = image[y:y+new_height, x:x+new_width]
                    augmented_images.append(cropped_image)
                else:
                    new_height = int(height / ratio)
                    new_width = int(width / ratio)
                    y = int((height - new_height) / 2)
                    x = int((width - new_width) / 2)
                    resized_image = cv2.resize(image, (new_width, new_height))
                    padded_image = cv2.copyMakeBorder(resized_image, y, y, x, x, cv2.BORDER_CONSTANT)
                    augmented_images.append(padded_image)
            for i, augmented_image in enumerate(augmented_images):
                augmented_file_name = f"augmented_{i}_{file_name}"
                augmented_file_path = os.path.join(root, augmented_file_name)
                cv2.imwrite(augmented_file_path, augmented_image)