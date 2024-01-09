import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import mobile_vit_xx_small as create_model


def classify_image(model, device, image_path, class_indict):
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    assert os.path.exists(image_path), "file: '{}' does not exist.".format(image_path)
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    class_name = class_indict[str(predict_cla)]
    probability = predict[predict_cla].numpy()
    return class_name, probability


def classify_folder(model, device, folder_path, class_indict):
    correct_count = 0
    total_count = 0
    predictions = []

    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            for file_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, file_name)
                class_name, _ = classify_image(model, device, image_path, class_indict)

                total_count += 1
                if class_name == dir_name:
                    correct_count += 1

                predictions.append({
                    'image_path': image_path,
                    'predicted_class': class_name,
                    'true_class': dir_name
                })

    accuracy = correct_count / total_count
    return predictions, accuracy


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=5).to(device)
    # load model weights
    model_weight_path = "/content/drive/MyDrive/城市工程系统智能化/MobileViT/weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    folder_path = "/content/drive/MyDrive/城市工程系统智能化/Soil types test"  # 修改为要分类的文件夹路径
    predictions, accuracy = classify_folder(model, device, folder_path, class_indict)

    print("Accuracy: {:.2%}".format(accuracy))

    for pred in predictions:
        print("Image: {}, Predicted class: {}, True class: {}".format(
            pred['image_path'], pred['predicted_class'], pred['true_class']))


if __name__ == '__main__':
    main()