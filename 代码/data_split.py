import os
import random
import shutil
# 源文件夹路径
source_folder = '/content/drive/MyDrive/城市工程系统智能化/Soil types'
# 目标文件夹路径
destination_folder1 = '/content/drive/MyDrive/城市工程系统智能化/Soil types train'
destination_folder2 = '/content/drive/MyDrive/城市工程系统智能化/Soil types test'
# 创建目标文件夹
os.makedirs(destination_folder1, exist_ok=True)
os.makedirs(destination_folder2, exist_ok=True)
# 遍历源文件夹中的每个文件夹
for folder_name in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder_name)
    if os.path.isdir(folder_path):
        # 创建目标子文件夹
        destination_subfolder1 = os.path.join(destination_folder1, folder_name)
        destination_subfolder2 = os.path.join(destination_folder2, folder_name)
        os.makedirs(destination_subfolder1, exist_ok=True)
        os.makedirs(destination_subfolder2, exist_ok=True)
        # 获取文件夹中的所有文件
        files = os.listdir(folder_path)
        # 计算训练集数量
        train_count = int(len(files) * 0.8)
        # 随机抽取训练集文件
        train_files = random.sample(files, train_count)
        # 将训练集文件复制到目标子文件夹中
        for file_name in train_files:
            file_path = os.path.join(folder_path, file_name)
            shutil.copy(file_path, destination_subfolder1)
        # 将剩余的文件作为测试集
        test_files = list(set(files) - set(train_files))
        # 将测试集文件复制到目标子文件夹中
        for file_name in test_files:
            file_path = os.path.join(folder_path, file_name)
            shutil.copy(file_path, destination_subfolder2)