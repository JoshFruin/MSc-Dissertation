# main.py

import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bars

# Imports within the project
from models import ViT, GNN, HybridModel

# Suppress all warnings globally
warnings.filterwarnings("ignore")

# Define paths to CSV files
csv_path_meta = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/meta.csv'
csv_path_dicom = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv'
mass_train_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv'
mass_test_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv'
calc_train_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_train_set.csv'
calc_test_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_test_set.csv'

# Read CSV files into DataFrames
df_meta = pd.read_csv(csv_path_meta)
dicom_data = pd.read_csv(csv_path_dicom)
mass_train_data = pd.read_csv(mass_train_path)
mass_test_data = pd.read_csv(mass_test_path)
calc_train_data = pd.read_csv(calc_train_path)
calc_test_data = pd.read_csv(calc_test_path)

# Define the image directory
image_dir = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/jpeg'


# Function to fix paths in the DataFrame
def fix_image_paths(df, image_dir):
    """
    Fix the paths in the DataFrame to point to the correct image directory.
    """
    return df.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))


dicom_data['image_path'] = fix_image_paths(dicom_data['image_path'], image_dir)

# Filter the DataFrame based on SeriesDescription
full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
roi_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path


# Helper function to create a dictionary from a Series of paths
def create_image_dict(image_series):
    """
    Create a dictionary from a Series of image paths.
    """
    image_dict = {}
    for dicom in image_series:
        key = dicom.split("/")[-2]
        image_dict[key] = dicom
    return image_dict


# Create dictionaries for each image type
full_mammogram_dict = create_image_dict(full_mammogram_images)
cropped_dict = create_image_dict(cropped_images)
roi_mask_dict = create_image_dict(roi_mask_images)


# Function to fix image paths in the dataset
def fix_image_path_mass(dataset, full_mammogram_dict, cropped_dict, roi_mask_dict):
    """
    Fix the image paths in the dataset based on the dictionaries.
    """
    valid_entries = []
    for i, img in enumerate(dataset.values):
        if len(img) > 11 and isinstance(img[11], str) and '/' in img[11]:
            img_name = img[11].split("/")[-2]
            if img_name in full_mammogram_dict:
                dataset.iloc[i, 11] = full_mammogram_dict[img_name]

        if len(img) > 12 and isinstance(img[12], str) and '/' in img[12]:
            img_name = img[12].split("/")[-2]
            if img_name in cropped_dict:
                dataset.iloc[i, 12] = cropped_dict[img_name]

        if len(img) > 13 and isinstance(img[13], str) and '/' in img[13]:
            img_name = img[13].split("/")[-2]
            if img_name in roi_mask_dict:
                dataset.iloc[i, 13] = roi_mask_dict[img_name]

        # Add to valid entries if full mammogram and mask are present
        if dataset.iloc[i, 11] in full_mammogram_dict.values() and dataset.iloc[i, 13] in roi_mask_dict.values():
            valid_entries.append(i)

    return dataset.iloc[valid_entries]


# Apply the function to the mass and calc datasets
mass_train_data = fix_image_path_mass(mass_train_data, full_mammogram_dict, cropped_dict, roi_mask_dict)
mass_test_data = fix_image_path_mass(mass_test_data, full_mammogram_dict, cropped_dict, roi_mask_dict)
calc_train_data = fix_image_path_mass(calc_train_data, full_mammogram_dict, cropped_dict, roi_mask_dict)
calc_test_data = fix_image_path_mass(calc_test_data, full_mammogram_dict, cropped_dict, roi_mask_dict)

# Rename columns for consistency
mass_train = mass_train_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                             'image view': 'image_view',
                                             'abnormality id': 'abnormality_id',
                                             'abnormality type': 'abnormality_type',
                                             'mass shape': 'mass_shape',
                                             'mass margins': 'mass_margins',
                                             'image file path': 'image_file_path',
                                             'cropped image file path': 'cropped_image_file_path',
                                             'ROI mask file path': 'ROI_mask_file_path'})

calc_train = calc_train_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                             'breast density': 'breast_density',
                                             'image view': 'image_view',
                                             'abnormality id': 'abnormality_id',
                                             'abnormality type': 'abnormality_type',
                                             'calc type': 'calc_type',
                                             'calc distribution': 'calc_distribution',
                                             'image file path': 'image_file_path',
                                             'cropped image file path': 'cropped_image_file_path',
                                             'ROI mask file path': 'ROI_mask_file_path'})

# Fill in missing values using the backwards fill method
mass_train['mass_shape'] = mass_train['mass_shape'].fillna(method='bfill')
mass_train['mass_margins'] = mass_train['mass_margins'].fillna(method='bfill')

# Map labels to integers
label_mapping = {'BENIGN': 0, 'MALIGNANT': 1}
mass_train['abnormality_type'] = mass_train['abnormality_type'].map(label_mapping)

# Combine mammogram and mask paths in one DataFrame
mam_train_data = mass_train[['image_file_path', 'ROI_mask_file_path', 'abnormality_type']].copy()
mam_train_data.columns = ['image_path', 'mask_path', 'label']


# Define custom Dataset
class BreastCancerDataset(Dataset):
    """
    Custom Dataset for breast cancer images and masks.
    """

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df['image_path'].values
        self.masks = df['mask_path'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image and mask
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        if self.transform:
            # Apply transformations
            image = self.transform(image)
            mask = self.transform(mask)

        # Get the label
        label = self.labels[idx]
        return image, mask, label


# Define transformations for the images and masks
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Create datasets and dataloaders
train_dataset = BreastCancerDataset(mam_train_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model components
vit = ViT()
gnn = GNN()

# Model architecture combining ViTs and GNNs
model = HybridModel(vit, gnn)

# Initialize loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train(model, train_loader, criterion, optimizer, num_epochs=25):
    """
    Train the hybrid model for a number of epochs.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, masks, labels in tqdm(train_loader):  # Add tqdm for progress bars
            # Move data to the appropriate device
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images, masks)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Compute average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs=25)
