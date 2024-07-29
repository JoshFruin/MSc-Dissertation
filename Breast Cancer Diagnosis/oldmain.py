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
from tqdm import tqdm, trange  # Import tqdm for progress bars
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
import multiprocessing
# Suppress the specific torchvision warning
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
# Check torch version and if a GPU is available on the device
from torch.utils.data import DataLoader, random_split
# Suppress all warnings globally
warnings.filterwarnings("ignore")
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, image_transform=None, mask_transform=None):
        self.data = dataframe
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Filter out rows with missing masks
        self.data = self.data.dropna(subset=['ROI_mask_file_path'])

        # Labels are already preprocessed to 0 and 1
        self.labels = torch.tensor(self.data['pathology'].values, dtype=torch.long)

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
            self.breast_density = pd.to_numeric(self.data['breast_density'], errors='coerce').fillna(0).astype(int)
        else:
            self.breast_density = pd.Series(np.zeros(len(self.data)), index=self.data.index)

    def process_calc_type(self):
        if 'calc_type' in self.data.columns:
            self.calc_type = pd.get_dummies(self.data['calc_type'], prefix='calc_type')
        else:
            self.calc_type = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                          columns=['calc_type_N/A'])

    def process_calc_distribution(self):
        if 'calc_distribution' in self.data.columns:
            self.calc_distribution = pd.get_dummies(self.data['calc_distribution'], prefix='calc_dist')
        else:
            self.calc_distribution = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                                  columns=['calc_dist_N/A'])

    def process_subtlety(self):
        if 'subtlety' in self.data.columns:
            self.subtlety = pd.to_numeric(self.data['subtlety'], errors='coerce').fillna(0).astype(int)
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
            self.abnormality_type = pd.get_dummies(self.data['abnormality_type'], prefix='abnormality')
        else:
            self.abnormality_type = pd.DataFrame(np.zeros((len(self.data), 2)), index=self.data.index,
                                                 columns=['abnormality_mass', 'abnormality_calcification'])

    def process_mass_shape(self):
        if 'mass_shape' in self.data.columns:
            self.mass_shape = pd.get_dummies(self.data['mass_shape'], prefix='shape')
        else:
            self.mass_shape = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                           columns=['shape_N/A'])

    def process_mass_margins(self):
        if 'mass_margins' in self.data.columns:
            self.mass_margins = pd.get_dummies(self.data['mass_margins'], prefix='margins')
        else:
            self.mass_margins = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                             columns=['margins_N/A'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_file_path']
        mask_name = self.data.iloc[idx]['ROI_mask_file_path']

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("RGB")

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        label = self.labels[idx]

        # Get numerical features
        numerical = torch.tensor(self.numerical_features.iloc[idx].values, dtype=torch.float)

        # Get categorical features
        categorical = torch.tensor(self.categorical_features.iloc[idx].values, dtype=torch.float)

        return image, mask, numerical, categorical, label

    def get_feature_dimensions(self):
        return self.numerical_features.shape[1], self.categorical_features.shape[1]

# Define the model
class MultimodalModel(nn.Module):
    def __init__(self, num_numerical_features, num_categorical_features):
        super(MultimodalModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Mask feature extractor
        self.mask_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mask_bn = nn.BatchNorm2d(64)
        self.mask_relu = nn.ReLU(inplace=True)
        self.mask_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_layers = nn.Sequential(
            self.mask_conv, self.mask_bn, self.mask_relu, self.mask_maxpool,
            self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Numerical and categorical feature processors
        self.num_dense = nn.Linear(num_numerical_features, 128)
        self.cat_dense = nn.Linear(num_categorical_features, 128)

        # Fully connected layers
        self.fc1 = nn.Linear(num_ftrs * 2 + 128 + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, image, mask, numerical, categorical):
        x_img = self.resnet(image)
        x_mask = self.mask_layers(mask).view(mask.size(0), -1)
        x_num = self.relu(self.num_dense(numerical))
        x_cat = self.relu(self.cat_dense(categorical))

        combined = torch.cat((x_img, x_mask, x_num, x_cat), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# Training Function
def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False) as pbar:
        for inputs, masks, numerical, categorical, labels in train_loader:
            inputs, masks, numerical, categorical, labels = inputs.to(device), masks.to(device), numerical.to(device), categorical.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, masks, numerical, categorical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.update(1)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, val_loader, criterion, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False) as pbar:
            for inputs, masks, numerical, categorical, labels in val_loader:
                inputs, masks, numerical, categorical, labels = inputs.to(device), masks.to(device), numerical.to(device), categorical.to(device), labels.to(device)

                outputs = model(inputs, masks, numerical, categorical)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.update(1)

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Training loop
def main():
    # Set up multiprocessing
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn', force=True)

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

    # Function to rename columns
    def rename_columns(df):
        return df.rename(columns={
            'left or right breast': 'left_or_right_breast',
            'image view': 'image_view',
            'abnormality id': 'abnormality_id',
            'abnormality type': 'abnormality_type',
            'mass shape': 'mass_shape',
            'mass margins': 'mass_margins',
            'calc type': 'calc_type',
            'calc distribution': 'calc_distribution',
            'image file path': 'image_file_path',
            'cropped image file path': 'cropped_image_file_path',
            'ROI mask file path': 'ROI_mask_file_path'
        })

    # Rename columns for all dataframes
    mass_train_data = rename_columns(mass_train_data)
    mass_test_data = rename_columns(mass_test_data)
    calc_train_data = rename_columns(calc_train_data)
    calc_test_data = rename_columns(calc_test_data)

    # Combine mass and calc data
    mam_train_data = pd.concat([mass_train_data, calc_train_data], ignore_index=True)
    mam_test_data = pd.concat([mass_test_data, calc_test_data], ignore_index=True)

    # Fill missing values for specific columns
    columns_to_fill = ['mass_shape', 'mass_margins', 'calc_type', 'calc_distribution']
    for col in columns_to_fill:
        if col in mam_train_data.columns:
            mam_train_data[col] = mam_train_data[col].fillna('N/A')
        if col in mam_test_data.columns:
            mam_test_data[col] = mam_test_data[col].fillna('N/A')

    # Map pathology labels
    label_mapping = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}
    mam_train_data['pathology'] = mam_train_data['pathology'].map(label_mapping)
    mam_test_data['pathology'] = mam_test_data['pathology'].map(label_mapping)

    # Drop rows with missing pathology
    mam_train_data = mam_train_data.dropna(subset=['pathology'])
    mam_test_data = mam_test_data.dropna(subset=['pathology'])

    # Convert pathology to int
    mam_train_data['pathology'] = mam_train_data['pathology'].astype(int)
    mam_test_data['pathology'] = mam_test_data['pathology'].astype(int)

    # Verify data
    print("Train data shape:", mam_train_data.shape)
    print("Test data shape:", mam_test_data.shape)
    print("\nTrain data columns:")
    print(mam_train_data.columns)
    print("\nMissing values in train data:")
    print(mam_train_data.isnull().sum())
    print("\nValue counts for 'pathology' in train data:")
    print(mam_train_data['pathology'].value_counts(normalize=True))
    def check_data_consistency(df, name):
        print(f"\nChecking {name} data consistency:")
        print("Unique values in 'abnormality_type':", df['abnormality_type'].unique())
        print("Rows with mass data:", df['mass_shape'].notna().sum())
        print("Rows with calc data:", df['calc_type'].notna().sum())
        print("Total rows:", len(df))

    check_data_consistency(mam_train_data, "train")
    check_data_consistency(mam_test_data, "test")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    # Define transforms for validation and testing
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create datasets with appropriate transforms
    train_dataset = BreastCancerDataset(mam_train_data, image_transform=train_transform,
                                        mask_transform=train_mask_transform)
    val_dataset = BreastCancerDataset(mam_train_data, image_transform=val_test_transform,
                                      mask_transform=val_test_mask_transform)
    test_dataset = BreastCancerDataset(mam_test_data, image_transform=val_test_transform,
                                       mask_transform=val_test_mask_transform)

    # Get the number of numerical and categorical features
    num_numerical_features, num_categorical_features = train_dataset.get_feature_dimensions()

    # Split train dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    _, val_dataset = random_split(val_dataset, [train_size, val_size])

    # Create dataloaders
    batch_size = 32
    num_workers = min(os.cpu_count(), 6)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalModel(num_numerical_features, num_categorical_features).to(device)

    patience = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    num_epochs = 10

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, num_epochs)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve += 1

        if no_improve == patience:
            print("Early stopping!")
            break

    """# Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")"""

if __name__ == '__main__':
    main()