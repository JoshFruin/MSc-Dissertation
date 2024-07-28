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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from PIL import Image
import random
from data_verification import (verify_data_linkage, verify_dataset_integrity,
                               check_mask_values, check_data_consistency,
                               check_label_consistency, visualize_augmented_samples,
                               verify_data_loading, verify_label_distribution,
                               verify_image_mask_correspondence, verify_batch, verify_labels) #check_data_range,
from models import ViTGNNHybrid, SimpleCNN, TransferLearningModel
import torchvision.models as models

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

"""# Data Visualization
visualisation_choice = int(input("Do you want to visualize the data? (1 for yes, 0 for no): "))
if visualisation_choice == 1:
    visualise_data(mass_train, calc_train)
else:
    print("Data visualisation skipped.")"""

class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None, mask_transform=None):
        self.data = dataframe
        self.transform = transform
        self.mask_transform = mask_transform

        # Filter out rows with missing masks
        self.data = self.data.dropna(subset=[self.data.columns[13]])  # Use the 14th column (index 13) for mask path

        # Map label values to integers
        self.labels = torch.tensor(
            self.data['pathology'].map({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}).fillna(0).astype(
                int).values, dtype=torch.long)

        # Process additional features
        self.process_breast_density()
        self.process_calc_type()
        self.process_calc_distribution()
        self.process_subtlety()
        self.process_left_or_right()
        self.process_abnormality_type()
        self.process_mass_shape()
        self.process_mass_margins()

        # Combine all numerical features
        self.numerical_features = pd.concat([self.subtlety, self.breast_density], axis=1)

        # Combine all categorical features
        self.categorical_features = pd.concat([
            self.calc_type, self.calc_distribution, self.left_or_right,
            self.abnormality_type, self.mass_shape, self.mass_margins
        ], axis=1)

    def process_breast_density(self):
        if 'breast_density' in self.data.columns:
            self.breast_density = self.data['breast_density'].map({'1': 0, '2': 1, '3': 2, '4': 3}).fillna(0).astype(int)
        else:
            self.breast_density = pd.Series(np.zeros(len(self.data)), index=self.data.index)

    def process_calc_type(self):
        if 'calc_type' in self.data.columns:
            calc_type_mapping = {
                'AMORPHOUS': 0, 'PLEOMORPHIC': 1, 'N/A': 2, 'ROUND_AND_REGULAR-LUCENT_CENTER-DYSTROPHIC': 3,
                'PUNCTATE': 4, 'COARSE': 5, 'VASCULAR': 6, 'FINE_LINEAR_BRANCHING': 7, 'LARGE_RODLIKE': 8,
                'ROUND_AND_REGULAR-EGGSHELL': 9, 'LUCENT_CENTER': 10, 'ROUND_AND_REGULAR': 11
            }
            self.calc_type = pd.get_dummies(self.data['calc_type'].map(calc_type_mapping).fillna(2), prefix='calc_type')
        else:
            self.calc_type = pd.DataFrame(np.zeros((len(self.data), 12)), index=self.data.index,
                                          columns=[f'calc_type_{i}' for i in range(12)])

    def process_calc_distribution(self):
        if 'calc_distribution' in self.data.columns:
            calc_dist_mapping = {
                'CLUSTERED': 0, 'LINEAR': 1, 'REGIONAL': 2, 'DIFFUSELY_SCATTERED': 3, 'SEGMENTAL': 4, 'N/A': 5
            }
            self.calc_distribution = pd.get_dummies(self.data['calc_distribution'].map(calc_dist_mapping).fillna(5), prefix='calc_dist')
        else:
            self.calc_distribution = pd.DataFrame(np.zeros((len(self.data), 6)), index=self.data.index,
                                                  columns=[f'calc_dist_{i}' for i in range(6)])

    def process_subtlety(self):
        if 'subtlety' in self.data.columns:
            self.subtlety = self.data['subtlety'].fillna(0).astype(int)
        else:
            self.subtlety = pd.Series(np.zeros(len(self.data)), index=self.data.index)

    def process_left_or_right(self):
        if 'left_or_right_breast' in self.data.columns:
            self.left_or_right = pd.get_dummies(self.data['left_or_right_breast'], prefix='breast')
        else:
            self.left_or_right = pd.DataFrame(np.zeros((len(self.data), 2)), index=self.data.index,
                                              columns=['breast_LEFT', 'breast_RIGHT'])

    def process_abnormality_type(self):
        if 'abnormality_type' in self.data.columns:
            abnormality_mapping = {'mass': 0, 'calcification': 1}
            self.abnormality_type = pd.get_dummies(self.data['abnormality_type'].map(abnormality_mapping).fillna(2), prefix='abnormality')
        else:
            self.abnormality_type = pd.DataFrame(np.zeros((len(self.data), 2)), index=self.data.index,
                                                 columns=['abnormality_mass', 'abnormality_calcification'])

    def process_mass_shape(self):
        if 'mass_shape' in self.data.columns:
            shape_mapping = {
                'IRREGULAR-ARCHITECTURAL_DISTORTION': 0, 'ARCHITECTURAL_DISTORTION': 1, 'OVAL': 2,
                'IRREGULAR': 3, 'LOBULATED-LYMPH_NODE': 4, 'LOBULATED': 5, 'FOCAL_ASYMMETRIC_DENSITY': 6,
                'LYMPH_NODE': 7, 'ROUND': 8, 'N/A': 9
            }
            self.mass_shape = pd.get_dummies(self.data['mass_shape'].map(shape_mapping).fillna(9), prefix='shape')
        else:
            self.mass_shape = pd.DataFrame(np.zeros((len(self.data), 10)), index=self.data.index,
                                           columns=[f'shape_{i}' for i in range(10)])

    def process_mass_margins(self):
        if 'mass_margins' in self.data.columns:
            margins_mapping = {
                'SPICULATED': 0, 'ILL_DEFINED': 1, 'CIRCUMSCRIBED': 2, 'OBSCURED': 3,
                'N/A': 4, 'MICROLOBULATED': 5, 'CIRCUMSCRIBED-ILL_DEFINED': 6
            }
            self.mass_margins = pd.get_dummies(self.data['mass_margins'].map(margins_mapping).fillna(4), prefix='margins')
        else:
            self.mass_margins = pd.DataFrame(np.zeros((len(self.data), 7)), index=self.data.index,
                                             columns=[f'margins_{i}' for i in range(7)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 11]  # Full mammogram image
        mask_name = self.data.iloc[idx, 13]  # ROI mask

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("RGB")  # Convert mask to RGB

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        label = self.labels[idx]

        # Get numerical features
        numerical = torch.tensor(self.numerical_features.iloc[idx].values, dtype=torch.float)

        # Get categorical features
        categorical = torch.tensor(self.categorical_features.iloc[idx].values, dtype=torch.float)

        return image, mask, numerical, categorical, label

# Create train and validation splits
train_df, val_df = train_test_split(mam_train_data, test_size=0.2, random_state=42)

# Print basic statistics of the dataset
print(train_df.describe())

# Check for class imbalance
print(train_df['pathology'].value_counts(normalize=True))

# Check for missing values
print(train_df.isnull().sum())

"""# Check the distribution of numerical features
for col in train_df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(train_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Check the distribution of categorical features
for col in train_df.select_dtypes(include=['object']).columns:
    print(train_df[col].value_counts(normalize=True))"""

# Define transformations with data augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Create datasets
train_dataset = BreastCancerDataset(train_df, transform=train_transform, mask_transform=mask_transform)
val_dataset = BreastCancerDataset(val_df, transform=val_transform, mask_transform=mask_transform)
from torch.utils.data import DataLoader, WeightedRandomSampler
# Compute class weights for balanced sampling
class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.labels),
                                     y=train_dataset.labels.numpy())
sample_weights = [class_weights[t] for t in train_dataset.labels.numpy()]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class MultimodalBreastCancerModel(nn.Module):
    def __init__(self, num_numerical_features, num_categorical_features):
        super(MultimodalBreastCancerModel, self).__init__()

        # Image processing
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()

        # Mask processing
        self.mask_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Numerical features processing
        self.numerical_fc = nn.Sequential(
            nn.Linear(num_numerical_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)
        )

        # Categorical features processing
        self.categorical_fc = nn.Sequential(
            nn.Linear(num_categorical_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3)
        )

        # Combine all features
        combined_features = 512 + 64 + 32 + 64  # ResNet18 output + mask output + numerical + categorical
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary classification
        )

    def forward(self, image, mask, numerical, categorical):
        image_features = self.resnet(image)
        mask_features = self.mask_conv(mask).view(mask.size(0), -1)
        numerical_features = self.numerical_fc(numerical)
        categorical_features = self.categorical_fc(categorical)

        combined = torch.cat((image_features, mask_features, numerical_features, categorical_features), dim=1)
        output = self.classifier(combined)
        return output

from torch.nn.utils import clip_grad_norm_

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks, numerical, categorical, labels = [item.to(device) for item in batch]

            optimizer.zero_grad()
            outputs = model(images, masks, numerical, categorical)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                images, masks, numerical, categorical, labels = [item.to(device) for item in batch]
                outputs = model(images, masks, numerical, categorical)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_accuracy = 100. * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

        scheduler.step(val_loss)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_multimodal_breast_cancer_model.pth')

    print(f'Best validation accuracy: {best_val_accuracy:.2f}%')
    return model

# Usage example
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_numerical_features = train_dataset.numerical_features.shape[1]
num_categorical_features = train_dataset.categorical_features.shape[1]

model = MultimodalBreastCancerModel(num_numerical_features, num_categorical_features).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Assuming you have already created train_loader and val_loader
num_epochs = 5
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)