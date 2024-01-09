import glob, json, os
import cv2
from PIL import Image
from tqdm import tqdm_notebook
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import requests
from transformers import CLIPProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
img_path = "/content/drive/MyDrive/城市工程系统智能化/Soil types/Black Soil/10.jpg"
image = Image.open(img_path)
display(image)
#inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
labels = ['blake soil', 'yellow soil', 'cinder soil', 'pear soil', 'laterite soil']
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
# 打印结果
for label, prob in zip(labels, probs.squeeze()):
    print('该图片为 %s 的概率是：%.02f%%' % (label, prob*100.))
max_prob_index = probs.argmax()
label = labels[max_prob_index]
print('该图片是%s' % label)