import os
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
# 定义文件夹路径
main_folder_path = "/content/drive/MyDrive/城市工程系统智能化/soil types train"
# 初始化 CLIP 处理器和模型
model_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
# 设置设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 定义图像预处理转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# 微调模型
model.train()
# 将类别映射到整数标签
class_to_index = {class_name: index for index, class_name in enumerate(os.listdir(main_folder_path))}
for subfolder_name in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder_name)
    # 跳过非文件夹的项目
    if not os.path.isdir(subfolder_path):
        continue
    # 为每个子文件夹中的图像加载数据并进行微调
    for image_name in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_name)
        # 加载图像并进行预处理
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[subfolder_name], images=transform(image).unsqueeze(0).to(device), return_tensors="pt")
        # 获取标签的编码
        label = torch.tensor([class_to_index[subfolder_name]]).to(device)
        # 前向传播
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = torch.nn.functional.softmax(logits_per_image[0], dim=0)
        # 计算损失
        loss = criterion(logits_per_image.view(1, -1), label)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 保存微调后的模型
    model.save_pretrained("/content/drive/MyDrive/城市工程系统智能化/models")