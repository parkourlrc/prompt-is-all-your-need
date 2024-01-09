import glob
import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
model_name = "openai/clip-vit-large-patch14"
pretrained_model_path = "/content/drive/MyDrive/城市工程系统智能化/models"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
data_dir = "/content/drive/MyDrive/城市工程系统智能化/Soil types test"
#少量测试集上最好的
labels = ['black soil', 'yellow soil', 'cinder soil', 'peat soil', 'laterite soil']
# Initialize counters
match_count = 0
mismatch_count = 0
# Iterate over subfolders
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        # Iterate over image files in the subfolder
        image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        for img_path in tqdm(image_files, desc=f"Processing {folder_name}"):
            image = Image.open(img_path)
            inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            max_prob_index = probs.argmax().item()
            predicted_label = labels[max_prob_index].lower()
            # Compare predicted label with folder name
            if predicted_label == folder_name.lower():
                match_count += 1
                label_match = "Y"
            else:
                mismatch_count += 1
                label_match = "N"
            print(f"Image: {img_path}")
            print(f"Predicted Label: {predicted_label}")
            print(f"Folder Name: {folder_name}")
            print(f"Label Match: {label_match}\n")
# Print the final counts
print(f"Match Count: {match_count}")
print(f"Mismatch Count: {mismatch_count}")
# Calculate classification metrics
total_samples = match_count + mismatch_count
accuracy = match_count / total_samples
error_rate = mismatch_count / total_samples
print(f"Accuracy: {accuracy:.2%}")
print(f"Error Rate: {error_rate:.2%}")