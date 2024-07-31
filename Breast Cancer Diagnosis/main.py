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
from models import SimpleCNN, TransferLearningModel, MultimodalModel
import torchvision.models as models
import multiprocessing
# Suppress the specific torchvision warning
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
# Check torch version and if a GPU is available on the device
from torch.utils.data import DataLoader, random_split
from data_preparation import fix_image_paths, create_image_dict, fix_image_path_mass, fix_image_path_calc, rename_columns
from utils import FocalLoss
# Suppress all warnings globally
warnings.filterwarnings("ignore")

def get_combined_categorical_columns(train_df, val_df, test_df, categorical_columns):
    combined_df = pd.concat([train_df[categorical_columns], val_df[categorical_columns], test_df[categorical_columns]], axis=0)
    combined_dummies = pd.get_dummies(combined_df, columns=categorical_columns, dummy_na=True)
    return combined_dummies.columns.tolist()

class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None, categorical_columns=None, all_categorical_columns=None):
        self.data = dataframe
        self.transform = transform
        self.categorical_columns = categorical_columns or ['calc_type', 'calc_distribution', 'left_or_right_breast',
                                                           'abnormality_type', 'mass_shape', 'mass_margins']
        self.all_categorical_columns = all_categorical_columns

        # Filter out rows with missing masks
        self.data = self.data.dropna(subset=['ROI_mask_file_path'])

        # Labels are already preprocessed to 0 and 1
        self.labels = torch.tensor(self.data['pathology'].values, dtype=torch.long)

        # Process numerical features
        self.numerical_features = self.data[['subtlety', 'breast_density']].fillna(0)

        # Process categorical features
        self.categorical_features = pd.get_dummies(self.data[self.categorical_columns], columns=self.categorical_columns, dummy_na=True)

        # Ensure all categorical features are present in the dataframe
        for col in self.all_categorical_columns:
            if col not in self.categorical_features:
                self.categorical_features[col] = 0

        # Reorder columns to match the combined dummies columns
        self.categorical_features = self.categorical_features[self.all_categorical_columns]

        # Ensure all categorical features are float
        self.categorical_features = self.categorical_features.astype(float)

        print(f"Shape of categorical features: {self.categorical_features.shape}")
        print(f"Categorical feature dtypes: {self.categorical_features.dtypes.value_counts()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_file_path']
        mask_name = self.data.iloc[idx]['ROI_mask_file_path']

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("RGB")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        label = self.labels[idx]

        numerical = torch.tensor(self.numerical_features.iloc[idx].values, dtype=torch.float)
        categorical = torch.tensor(self.categorical_features.iloc[idx].values, dtype=torch.float)

        return image, mask, numerical, categorical, label

    def get_num_features(self):
        num_numerical_features = self.numerical_features.shape[1]
        num_categorical_features = self.categorical_features.shape[1]
        return num_numerical_features, num_categorical_features


    @staticmethod
    def get_feature_dimensions(train_df, val_df, test_df, categorical_columns=None):
        categorical_columns = categorical_columns or ['calc_type', 'calc_distribution', 'left_or_right_breast',
                                                      'abnormality_type', 'mass_shape', 'mass_margins']

        # Combine train, validation, and test data for categorical encoding
        combined_df = pd.concat([train_df[categorical_columns], val_df[categorical_columns], test_df[categorical_columns]], axis=0)

        # Get dummies for all possible categories
        all_categories = pd.get_dummies(combined_df, columns=categorical_columns, dummy_na=True)

        num_numerical_features = 2  # subtlety and breast_density
        num_categorical_features = all_categories.shape[1]

        print(f"Number of categorical features after combining train, validation, and test: {num_categorical_features}")

        return num_numerical_features, num_categorical_features, all_categories.columns.tolist()

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


def test_model(model, test_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, masks, numerical, categorical, labels in tqdm(test_loader, desc="Testing"):
            inputs, masks, numerical, categorical, labels = inputs.to(device), masks.to(device), numerical.to(
                device), categorical.to(device), labels.to(device)

            outputs = model(inputs, masks, numerical, categorical)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc_roc = roc_auc_score(all_labels, all_preds)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print results
    print(f"Test Loss: {running_loss / len(test_loader):.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return accuracy, precision, recall, f1, auc_roc
from torchvision.transforms import ToPILImage, ToTensor
class AlignedTransform:
    def __init__(self, size=(224, 224), flip_prob=0.5, rotate_prob=0.5, max_rotation=10):
        self.size = size
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_rotation = max_rotation
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, mask):
        # Convert to PIL Image if tensor
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        if isinstance(mask, torch.Tensor):
            mask = TF.to_pil_image(mask)

        # Ensure image is grayscale
        image = image.convert("L")

        # Resize
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size)

        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random rotation
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # Convert to tensor
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        # Normalize (only for image)
        image = self.normalize(image)

        return image, mask

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

    # Apply the function to the image paths
    dicom_data['image_path'] = fix_image_paths(dicom_data['image_path'], image_dir)

    # Filter the DataFrame based on SeriesDescription
    full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
    cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
    roi_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path

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

    # Apply the function to the mass and calc datasets
    mass_train_data = fix_image_path_mass(mass_train_data, full_mammogram_dict, cropped_dict, roi_mask_dict)
    mass_test_data = fix_image_path_mass(mass_test_data, full_mammogram_dict, cropped_dict, roi_mask_dict)

    # Check the updated DataFrames (optional, for checking)
    print(mass_train_data.head())
    print(mass_test_data.head())

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

    # Split the combined training data into train and validation sets
    mam_train_data, mam_val_data = train_test_split(mam_train_data, test_size=0.2, random_state=42,
                                            stratify=mam_train_data['pathology'])

    # Define transforms
    train_transform = AlignedTransform(size=(224, 224), flip_prob=0.5, rotate_prob=0.5, max_rotation=10)
    val_test_transform = AlignedTransform(size=(224, 224), flip_prob=0, rotate_prob=0)

    # Assuming you have three DataFrames: train_df, val_df, and test_df for your datasets
    num_numerical_features, num_categorical_features, categorical_feature_columns = BreastCancerDataset.get_feature_dimensions(
        mam_train_data, mam_val_data, mam_test_data)

    # Assuming you have three DataFrames: train_df, val_df, and test_df for your datasets
    categorical_columns = ['calc_type', 'calc_distribution', 'left_or_right_breast', 'abnormality_type', 'mass_shape',
                           'mass_margins']
    all_categorical_columns = get_combined_categorical_columns(mam_train_data, mam_val_data, mam_test_data,
                                                               categorical_columns)

    print(f"Number of categorical features: {len(all_categorical_columns)}")

    # Now you can initialize your datasets
    train_dataset = BreastCancerDataset(mam_train_data, transform=train_transform, categorical_columns=categorical_columns, all_categorical_columns=all_categorical_columns)
    val_dataset = BreastCancerDataset(mam_val_data, transform=val_test_transform, categorical_columns=categorical_columns, all_categorical_columns=all_categorical_columns)
    test_dataset = BreastCancerDataset(mam_test_data, transform=val_test_transform, categorical_columns=categorical_columns, all_categorical_columns=all_categorical_columns)
    # Visualise Augmentations
    def visualize_augmentations(dataset, num_samples=5):
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4 * num_samples))

        for i in range(num_samples):
            idx = np.random.randint(len(dataset))
            original_image, original_mask, _, _, _ = dataset[idx]

            # Apply augmentation
            augmented_image, augmented_mask = dataset.transform(original_image, original_mask)

            # Convert to numpy arrays and prepare for display
            def prepare_for_display(img):
                if img.shape[0] == 1:  # If grayscale
                    img = img.squeeze(0)
                else:  # If RGB
                    img = img.permute(1, 2, 0)
                img = img.numpy()
                img = (img - img.min()) / (img.max() - img.min())
                return img

            original_image = prepare_for_display(original_image)
            augmented_image = prepare_for_display(augmented_image)
            original_mask = prepare_for_display(original_mask)
            augmented_mask = prepare_for_display(augmented_mask)

            # Display images
            axes[i, 0].imshow(original_image, cmap='gray')
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(original_mask, cmap='gray')
            axes[i, 1].set_title('Original Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(augmented_image, cmap='gray')
            axes[i, 2].set_title('Augmented Image')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(augmented_mask, cmap='gray')
            axes[i, 3].set_title('Augmented Mask')
            axes[i, 3].axis('off')

            # Print applied transformations
            print(f"Sample {i + 1} transformations:")
            print(f"  Flip: {'Yes' if np.random.rand() < dataset.transform.flip_prob else 'No'}")
            print(f"  Rotate: {'Yes' if np.random.rand() < dataset.transform.rotate_prob else 'No'}")
            print(f"  Normalization: Applied to image")
            print()

        plt.tight_layout()
        plt.show()

        """# Print debug information
        print(f"Original image shape: {original_image.shape}, dtype: {original_image.dtype}")
        print(f"Augmented image shape: {augmented_image.shape}, dtype: {augmented_image.dtype}")
        print(f"Original numpy shape: {original_image_np.shape}, dtype: {original_image_np.dtype}")
        print(f"Augmented numpy shape: {augmented_image_np.shape}, dtype: {augmented_image_np.dtype}")"""
    # Usage
    visualize_augmentations(train_dataset)

    def verify_augmentations(dataset, num_samples=10):
        flip_count = 0
        rotate_count = 0
        total_brightness_change = 0
        total_contrast_change = 0
        total_saturation_change = 0

        for _ in range(num_samples):
            idx = np.random.randint(len(dataset))
            original_image, original_mask, _, _, _ = dataset[idx]
            augmented_image, augmented_mask = dataset.transform(original_image, original_mask)

            # Check for flip
            if torch.all(augmented_image.flip(dims=[2]) == original_image):
                flip_count += 1

            # Check for rotation (this is an approximation)
            if not torch.allclose(augmented_image, original_image, atol=1e-3):
                rotate_count += 1

            # Check color jitter
            orig_brightness = original_image.mean()
            aug_brightness = augmented_image.mean()
            total_brightness_change += abs(aug_brightness - orig_brightness)

            orig_contrast = original_image.std()
            aug_contrast = augmented_image.std()
            total_contrast_change += abs(aug_contrast - orig_contrast)

            orig_saturation = original_image.std(dim=0).mean()
            aug_saturation = augmented_image.std(dim=0).mean()
            total_saturation_change += abs(aug_saturation - orig_saturation)

        print(f"Flip probability: {flip_count / num_samples:.2f}")
        print(f"Rotation probability: {rotate_count / num_samples:.2f}")
        print(f"Average brightness change: {total_brightness_change / num_samples:.4f}")
        print(f"Average contrast change: {total_contrast_change / num_samples:.4f}")
        print(f"Average saturation change: {total_saturation_change / num_samples:.4f}")

    # Usage
    verify_augmentations(train_dataset)

    # Create dataloaders
    batch_size = 32
    num_workers = min(os.cpu_count(), 6)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MultimodalModel(num_numerical_features, num_categorical_features).to(device)

    patience = 5
    #criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    num_epochs = 10

    best_val_loss = float('inf')
    no_improve = 0

    # Training loop
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

    # Load best model and evaluate on test set
    print("Loading best model and evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    accuracy, precision, recall, f1, auc_roc = test_model(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()