# Imports
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm  # Import tqdm for progress bars
from data_visualisations import visualise_data, dataloader_visualisations
from torch_geometric.data import Data, Batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image
import random
from data_verification import (verify_data_linkage, verify_dataset_integrity,
                               check_mask_values, check_data_consistency,
                               check_label_consistency, visualize_augmented_samples,
                               verify_data_loading, verify_label_distribution,
                               verify_image_mask_correspondence, verify_batch, verify_labels) #check_data_range,
from models import ViTGNNHybrid, SimpleCNN, TransferLearningModel,MultimodalBreastCancerModel

import multiprocessing
from multiprocessing import freeze_support
# Suppress all warnings globally
warnings.filterwarnings("ignore")
# Check torch version and if a GPU is available on the device
print(torch.__version__)
print(torch.cuda.is_available())

"""
For the mammograms we're using the CBIS-DDSM dataset from Kaggle: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data
"""

# Provide the correct path to the CSV files
csv_path_meta = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/meta.csv'

csv_path_dicom = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv'

# Read the CSV files into DataFrames
df_meta = pd.read_csv(csv_path_meta)
dicom_data = pd.read_csv(csv_path_dicom)

# Read the CSV files for mass data
mass_train_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv'
mass_test_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv'

mass_train_data = pd.read_csv(mass_train_path)
mass_test_data = pd.read_csv(mass_test_path)

# Read the CSV files for calc data
calc_train_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_train_set.csv'
calc_test_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_test_set.csv'

calc_train_data = pd.read_csv(calc_train_path)
calc_test_data = pd.read_csv(calc_test_path)

# Display the DataFrames (optional, for checking)
print(df_meta.head())
print(dicom_data.head())

# Define the image directory
image_dir = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/jpeg'

"""Data Cleaning & Preprocessing"""

# Function to fix paths in the DataFrame
def fix_image_paths(df, image_dir):
    return df.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))

# Apply the function to the image paths
dicom_data['image_path'] = fix_image_paths(dicom_data['image_path'], image_dir)

# Filter the DataFrame based on SeriesDescription
full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
roi_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path

# Helper function to create a dictionary from a Series of paths
def create_image_dict(image_series):
    image_dict = {}
    for dicom in image_series:
        key = dicom.split("/")[-2]  # Assuming the key is the folder name before the image name
        image_dict[key] = dicom
    return image_dict

# Create dictionaries for each image type
full_mammogram_dict = create_image_dict(full_mammogram_images)
cropped_dict = create_image_dict(cropped_images)
roi_mask_dict = create_image_dict(roi_mask_images)

# Display a sample item from each dictionary (optional, for checking)
print(next(iter(full_mammogram_dict.items())))
print(next(iter(cropped_dict.items())))
print(next(iter(roi_mask_dict.items())))
print(mass_train_data.head())
print(mass_test_data.head())
print(calc_train_data.head())
print(calc_test_data.head())

# Define the path fixing function for the metadata DataFrames
def fix_image_path_mass(dataset, full_mammogram_dict, cropped_dict, roi_mask_dict):
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

# Check the updated DataFrames (optional, for checking)
print(mass_train_data.head())
print(mass_test_data.head())

def fix_image_path_calc(dataset, full_mammogram_dict, cropped_dict, roi_mask_dict):
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

# Apply the function to the calc datasets
calc_train_data = fix_image_path_calc(calc_train_data, full_mammogram_dict, cropped_dict, roi_mask_dict)
calc_test_data = fix_image_path_calc(calc_test_data, full_mammogram_dict, cropped_dict, roi_mask_dict)

# Check the updated DataFrames (optional, for checking)
print(calc_train_data.head())
print(calc_test_data.head())

# check unique values in pathology column
mass_train_data.pathology.unique()

calc_train_data.pathology.unique()

mass_train_data.info()

calc_train_data.info()

# rename columns
mass_train = mass_train_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

print(mass_train.head())

# rename columns
calc_train = calc_train_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                             'breast density':'breast_density',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'calc type': 'calc_type',
                                           'calc distribution': 'calc_distribution',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

print(calc_train.head())

# check for null values
print(mass_train.isnull().sum())
print(calc_train.isnull().sum())

# fill in missing values using the backwards fill method
mass_train['mass_shape'] = mass_train['mass_shape'].fillna(method='bfill')
mass_train['mass_margins'] = mass_train['mass_margins'].fillna(method='bfill')

#check null values
print(mass_train.isnull().sum())

# fill in missing values using the backwards fill method
calc_train['calc_type'] = calc_train['calc_type'].fillna(method='bfill')
calc_train['calc_distribution'] = calc_train['calc_distribution'].fillna(method='bfill')

#check null values
print(calc_train.isnull().sum())
print(mass_test_data.isnull().sum())
print(calc_test_data.isnull().sum())

# check for column names in mass_test
print(mass_test_data.columns)
print('\n')
# rename columns
mass_test = mass_test_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

# view renamed columns
print(mass_test.columns)

# check for column names in mass_test
print(calc_test_data.columns)
print('\n')
# rename columns
calc_test = calc_test_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'breast density':'breast_density',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'calc type': 'calc_type',
                                           'calc distribution': 'calc_distribution',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

# view renamed columns
print(calc_test.columns)

# fill in missing values using the backwards fill method
calc_test['calc_type'] = calc_test['calc_type'].fillna(method='bfill')
calc_test['calc_distribution'] = calc_test['calc_distribution'].fillna(method='bfill')
#check null values
print(calc_test.isnull().sum())

mass_train['pathology'].value_counts().plot(kind='bar')
calc_train['pathology'].value_counts().plot(kind='bar')

label_mapping = {'BENIGN': 0,'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}
mass_train['pathology'] = mass_train['pathology'].map(label_mapping)
mass_test['pathology'] = mass_test['pathology'].map(label_mapping)
calc_train['pathology'] = calc_train['pathology'].map(label_mapping)
calc_test['pathology'] = calc_test['pathology'].map(label_mapping)

print(mass_train['pathology'].value_counts())
print(mass_test['pathology'].value_counts())
print(calc_train['pathology'].value_counts())
print(calc_test['pathology'].value_counts())

mass_train = mass_train.dropna(subset=['pathology'])
mass_test = mass_test.dropna(subset=['pathology'])
calc_train = calc_train.dropna(subset=['pathology'])
calc_test = calc_test.dropna(subset=['pathology'])

mass_train['pathology'] = mass_train['pathology'].astype(int)
mass_test['pathology'] = mass_test['pathology'].astype(int)
calc_train['pathology'] = calc_train['pathology'].astype(int)
calc_test['pathology'] = calc_test['pathology'].astype(int)

# Combine mass_train_data and calc_train_data into mam_train_data
mam_train_data = pd.concat([mass_train_data, calc_train_data], ignore_index=True)

# Combine mass_test_data and calc_test_data into mam_test_data
mam_test_data = pd.concat([mass_test_data, calc_test_data], ignore_index=True)

# Data Visualization
"""visualisation_choice = int(input("Do you want to visualize the data? (1 for yes, 0 for no): "))
if visualisation_choice == 1:
    visualise_data(mass_train, calc_train)
else:
    print("Data visualisation skipped.")
"""
# Update BreastCancerDataset constructor to print unique values in 'pathology' column before mapping
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, image_transform=None, mask_transform=None):
        self.dataframe = dataframe
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Debugging: print column names and a few rows
        print("Columns in DataFrame:", self.dataframe.columns)
        print("Sample DataFrame Rows:\n", self.dataframe.head())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Debugging: print index and corresponding row
        print(f"Fetching item at index {idx}")
        print("Row Data:\n", self.dataframe.iloc[idx])

        img_path = self.dataframe.iloc[idx]['image_file_path']
        mask_path = self.dataframe.iloc[idx]['ROI_mask_file_path']
        label = self.dataframe.iloc[idx]['pathology']

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        tabular_data = torch.tensor([
            self.dataframe.iloc[idx]['breast_density'],
            # Add other relevant features here
        ], dtype=torch.float32)

        return image, mask, tabular_data, label

# Define transforms
# Image transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mask transforms
mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Test transforms (without augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Initialize datasets and dataloaders
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get the indices for train and validation sets
train_idx, val_idx = next(gss.split(mam_train_data, groups=mam_train_data['patient_id']))

# Split the data
train_data = mam_train_data.iloc[train_idx]
val_data = mam_train_data.iloc[val_idx]
print(mam_train_data.columns)
print(mam_test_data.columns)
print(val_data.columns)
print(train_data.columns)

train_dataset = BreastCancerDataset(train_data, image_transform=image_transform, mask_transform=mask_transform)
val_dataset = BreastCancerDataset(val_data, image_transform=test_transform, mask_transform=mask_transform)
test_dataset = BreastCancerDataset(mam_test_data, image_transform=test_transform, mask_transform=mask_transform)

"""print("Train set class distribution:", np.unique(train_dataset.labels, return_counts=True))
print("Validation set class distribution:", np.unique(val_dataset.labels, return_counts=True))
print("Test set class distribution:", np.unique(test_dataset.labels, return_counts=True))"""

# Modify the DataLoader to handle the graph data
def collate_fn(batch):
    images, masks, labels = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    labels = torch.tensor(labels)
    return images, masks, labels

from torch.utils.data import WeightedRandomSampler

"""def create_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

train_sampler = create_weighted_sampler(train_dataset.labels)"""

"""train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)"""
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
# Call dataloader_visualisations functions
"""visualisation_choice_2 = int(input("Do you want to visualise the dataloader? (1 for yes, 0 for no): "))
if visualisation_choice_2 == 1:
    dataloader_visualisations(train_dataset, test_dataset, train_loader, test_loader)
else:
    print("Dataloader visualisation skipped.")"""

# Call verification functions
"""verification_choice = int(input("Do you want to verify the data? (1 for yes, 0 for no): "))
if verification_choice == 1:
    from data_verification import (verify_data_linkage, verify_dataset_integrity,
                                   check_mask_values, check_data_consistency,
                                   check_label_consistency, visualize_augmented_samples,
                                   verify_data_loading, verify_label_distribution,
                                   verify_image_mask_correspondence, verify_batch, verify_labels) #check_data_range,

    print("Verifying mass training data linkage...")
    verify_data_linkage(mass_train_path, mass_train, full_mammogram_dict, roi_mask_dict)

    print("Verifying calc training data linkage...")
    verify_data_linkage(calc_train_path, calc_train, full_mammogram_dict, roi_mask_dict)

    print("Verifying mass test data linkage...")
    verify_data_linkage(mass_test_path, mass_test, full_mammogram_dict, roi_mask_dict)

    print("Verifying calc test data linkage...")
    verify_data_linkage(calc_test_path, calc_test, full_mammogram_dict, roi_mask_dict)

    verify_dataset_integrity(train_dataset)
    verify_dataset_integrity(val_dataset)
    verify_dataset_integrity(test_dataset)

    #check_data_range(train_dataset)
    check_mask_values(train_dataset)
    check_data_consistency(train_dataset, val_dataset, test_dataset)
    check_label_consistency(train_dataset)

    visualize_augmented_samples(train_dataset, image_transform)

    print("Verifying training data...")
    verify_data_loading(train_dataset)
    verify_label_distribution(train_dataset)
    verify_image_mask_correspondence(train_dataset)

    print("\nVerifying validation data...")
    verify_data_loading(val_dataset)
    verify_label_distribution(val_dataset)
    verify_image_mask_correspondence(val_dataset)

    print("\nVerifying test data...")
    verify_data_loading(test_dataset)
    verify_label_distribution(test_dataset)
    verify_image_mask_correspondence(test_dataset)

    verify_batch(train_loader)
    verify_labels(train_dataset)
else:
    print("Data verification skipped.")"""

"""# Initialize the ViT-GNN hybrid model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ViTGNNHybrid(num_classes=2).to(device)
#model = SimpleCNN(num_classes=2).to(device)
#model = TransferLearningModel(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
num_epochs = 5

# Define optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Initialize lists to store metrics for plotting
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []"""

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, masks, tabular_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            tabular_data = tabular_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, tabular_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return model

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, masks, tabular_data, labels in data_loader:
            images = images.to(device)
            tabular_data = tabular_data.to(device)
            labels = labels.to(device)

            outputs = model(images, tabular_data)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total

    return avg_loss, accuracy


def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, masks, tabular_data, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            tabular_data = tabular_data.to(device)
            labels = labels.to(device)

            outputs = model(images, tabular_data)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Calculate additional metrics
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    # ROC curve and AUC score (for binary classification)
    if len(set(all_labels)) == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_predictions)
        roc_auc = roc_auc_score(all_labels, all_predictions)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    return test_loss, test_acc
if __name__ == '__main__':
    freeze_support()
    multiprocessing.set_start_method('spawn')

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, criterion, optimizer, and scheduler
    model = MultimodalBreastCancerModel(num_tabular_features=1).to(device)  # Adjust num_tabular_features as needed
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Train the model
    num_epochs = 5
    print("Training...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

    # Test the model
    test_loss, test_acc = test_model(trained_model, test_loader, criterion, device)

"""
# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for images, masks, labels in data_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = F.softmax(outputs, dim=1)[:, 1]  # Get probabilities for the positive class
            all_preds.extend(preds.cpu().numpy())  # No need for detach() here as we're already in no_grad context
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    precision = precision_score(all_labels, (all_preds > 0.5).astype(int), average='weighted')
    recall = recall_score(all_labels, (all_preds > 0.5).astype(int), average='weighted')
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average='weighted')
    auc = roc_auc_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, (all_preds > 0.5).astype(int))

    return avg_loss, accuracy, precision, recall, f1, auc, cm, all_preds, all_labels

# Training loop
print("Training...")
best_val_loss = float('inf')
patience = 10
epochs_without_improvement = 0

# Initialize lists to store metrics for plotting
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []

    for images, masks, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        with torch.no_grad():
            preds = F.softmax(outputs, dim=1)[:, 1]
            train_preds.extend(preds.cpu().detach().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader)
    train_preds = np.array(train_preds)
    train_labels = np.array(train_labels)
    train_acc = accuracy_score(train_labels, (train_preds > 0.5).astype(int))

    # Validation
    val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_cm, _, _ = evaluate(model, val_loader, device)

    # Step the scheduler with the validation loss
    scheduler.step(val_loss)

    # Store metrics for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Print metrics
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
    print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')
    print("Validation Confusion Matrix:")
    print(val_cm)
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Early stopping and model saving
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping")
            break

print('Finished Training')

# Plot learning curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves - Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Curves - Accuracy')

plt.tight_layout()
plt.savefig('learning_curves.png')
plt.close()"""