import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import ViTModel, ViTFeatureExtractor

# Vision Transformer Feature Extractor and Model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Transform function for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Custom Dataset Class for Breast Cancer Data
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        self.labels = torch.tensor(self.data['pathology'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 11]  # Path to the full mammogram image
        if pd.isnull(img_name):
            return None
        image = Image.open(img_name).convert("RGB")  # Open image and convert to RGB
        if self.transform:
            image = self.transform(image)  # Apply transformations
        inputs = feature_extractor(images=image, return_tensors="pt")  # Prepare inputs for ViT
        vit_outputs = vit_model(**inputs)  # Get features from ViT
        features = vit_outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
        label = self.labels[idx]
        return features, label
