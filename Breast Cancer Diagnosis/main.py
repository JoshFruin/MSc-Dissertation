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
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bars

# Imports within the project
from data_visualisations import visualise_data, dataloader_visualisations
from models import ViTGNNHybrid, SimpleCNN
from torch_geometric.data import Data, Batch

# Suppress all warnings globally
warnings.filterwarnings("ignore")

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

import pandas as pd
import os
from PIL import Image
import random
import matplotlib.pyplot as plt


def comprehensive_verify_data_linkage(original_csv_path, processed_df, full_mammogram_dict, roi_mask_dict,
                                      num_samples=5):
    """
    Verify that images, masks, and labels are correctly linked from original CSV through preprocessing.

    Args:
    original_csv_path (str): Path to the original CSV file.
    processed_df (pd.DataFrame): The processed dataframe containing image paths and labels.
    full_mammogram_dict (dict): Dictionary mapping keys to full mammogram image paths.
    roi_mask_dict (dict): Dictionary mapping keys to ROI mask image paths.
    num_samples (int): Number of random samples to verify.

    Returns:
    None. Prints verification results and displays images.
    """
    print(f"Verifying {num_samples} random samples...")

    # Load original CSV
    original_df = pd.read_csv(original_csv_path)

    for i in range(num_samples):
        # Randomly select a row from the processed dataframe
        processed_row = processed_df.sample(n=1).iloc[0]

        # Find the corresponding row in the original CSV
        original_row = original_df[original_df['patient_id'] == processed_row['patient_id']].iloc[0]

        # Extract relevant information
        original_img_path = original_row['image file path']
        original_mask_path = original_row['ROI mask file path']
        processed_img_path = processed_row['image_file_path']
        processed_mask_path = processed_row['ROI_mask_file_path']
        label = processed_row['pathology']

        # Verify file paths
        img_key = processed_img_path.split("/")[-2]
        mask_key = processed_mask_path.split("/")[-2]

        if full_mammogram_dict[img_key] != processed_img_path:
            print(f"Error: Mismatch in image file path for {img_key}")
        if roi_mask_dict[mask_key] != processed_mask_path:
            print(f"Error: Mismatch in mask file path for {mask_key}")

        # Verify file existence
        if not os.path.exists(processed_img_path):
            print(f"Error: Image file does not exist: {processed_img_path}")
            continue
        if not os.path.exists(processed_mask_path):
            print(f"Error: Mask file does not exist: {processed_mask_path}")
            continue

        # Load image and mask
        try:
            image = Image.open(processed_img_path).convert("L")
            mask = Image.open(processed_mask_path).convert("L")
        except Exception as e:
            print(f"Error loading image or mask: {e}")
            continue

        # Display image, mask, and label
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(image, cmap='gray')
        ax1.set_title("Image")
        ax1.axis('off')

        ax2.imshow(mask, cmap='gray')
        ax2.set_title("Mask")
        ax2.axis('off')

        ax3.imshow(image)
        ax3.imshow(mask, alpha=0.5, cmap='jet')
        ax3.set_title("Image with Mask Overlay")
        ax3.axis('off')

        plt.suptitle(f"Sample {i + 1}: Label = {label}")
        plt.tight_layout()
        plt.show()

        print(f"Sample {i + 1}:")
        print(f"  Original Image Path: {original_img_path}")
        print(f"  Processed Image Path: {processed_img_path}")
        print(f"  Original Mask Path: {original_mask_path}")
        print(f"  Processed Mask Path: {processed_mask_path}")
        print(f"  Label: {label}")
        print("\n")

"""print("Verifying mass training data linkage...")
comprehensive_verify_data_linkage(
    mass_train_path,
    mass_train,
    full_mammogram_dict,
    roi_mask_dict
)

print("Verifying calc training data linkage...")
comprehensive_verify_data_linkage(
    calc_train_path,
    calc_train,
    full_mammogram_dict,
    roi_mask_dict
)

print("Verifying mass test data linkage...")
comprehensive_verify_data_linkage(
    mass_test_path,
    mass_test,
    full_mammogram_dict,
    roi_mask_dict
)

print("Verifying calc test data linkage...")
comprehensive_verify_data_linkage(
    calc_test_path,
    calc_test,
    full_mammogram_dict,
    roi_mask_dict
)
"""
# Data Visualization
"""visualisation_choice = int(input("Do you want to visualize the data? (1 for yes, 0 for no): "))
if visualisation_choice == 1:
    visualise_data(mass_train, calc_train)
else:
    print("Data visualisation skipped.")"""

# Update BreastCancerDataset constructor to print unique values in 'pathology' column before mapping
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None, mask_transform=None, threshold_mask=False):
        self.data = dataframe
        self.transform = transform
        self.mask_transform = mask_transform
        self.threshold_mask = threshold_mask

        # Filter out rows with missing masks
        self.data = self.data.dropna(subset=[self.data.columns[13]])  # Use the 14th column (index 13) for mask path

        # Map label values to integers
        self.labels = torch.tensor(
            self.data['pathology'].map({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}).fillna(0).astype(
                int).values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 11]  # Full mammogram image
        mask_name = self.data.iloc[idx, 13]  # ROI mask

        image = Image.open(img_name).convert("L")  # Ensure image is grayscale
        mask = Image.open(mask_name).convert("L")  # Ensure mask is grayscale

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        if self.threshold_mask:
            mask = (mask > 0.5).float()  # Threshold mask to binary values

        label = self.labels[idx]

        return image, mask, label

def verify_dataset_integrity(dataset, num_samples=5):
    print(f"Verifying {num_samples} random samples from the dataset...")
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask, label = dataset[idx]

        print(f"Sample {i + 1}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Label: {label}")

        # Check if image and mask have the same dimensions
        assert image.shape == mask.shape, f"Image and mask shapes do not match for sample {i + 1}"

        # Check if label is within expected range
        assert label in [0, 1], f"Unexpected label value {label} for sample {i + 1}"

        # Visualize the sample
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title("Image")
        plt.subplot(132)
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title("Mask")
        plt.subplot(133)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.imshow(mask.squeeze(), alpha=0.5, cmap='jet')
        plt.title("Overlay")
        plt.suptitle(f"Sample {i + 1}: Label = {label}")
        plt.show()

    print("Dataset integrity verification completed.")

# Define transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: x / 255.0),  # Scale to [0, 1]
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Lambda(lambda x: x / x.max())  # Normalize to [0,1]
])
mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize datasets and dataloaders
from sklearn.model_selection import GroupShuffleSplit

# Assuming 'patient_id' is a column in your DataFrame
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get the indices for train and validation sets
train_idx, val_idx = next(gss.split(mam_train_data, groups=mam_train_data['patient_id']))

# Split the data
train_data = mam_train_data.iloc[train_idx]
val_data = mam_train_data.iloc[val_idx]

train_dataset = BreastCancerDataset(dataframe=train_data, transform=image_transform, mask_transform=mask_transform)
print(train_dataset.data.columns)
val_dataset = BreastCancerDataset(dataframe=val_data, transform=image_transform, mask_transform=mask_transform)
test_dataset = BreastCancerDataset(dataframe=mam_test_data, transform=image_transform, mask_transform=mask_transform)

verify_dataset_integrity(train_dataset)
verify_dataset_integrity(val_dataset)
verify_dataset_integrity(test_dataset)
print("Train set class distribution:", np.unique(train_dataset.labels, return_counts=True))
print("Validation set class distribution:", np.unique(val_dataset.labels, return_counts=True))
print("Test set class distribution:", np.unique(test_dataset.labels, return_counts=True))

def check_data_range(dataset):
    """
    Check if the image and mask pixel values are within the expected range.
    """
    for i in range(len(dataset)):
        image, mask, _ = dataset[i]
        if not (image.min() >= 0 and image.max() <= 1):
            print(f"Image {i} has values outside [0, 1] range:")
            print(f"  Min: {image.min().item():.4f}, Max: {image.max().item():.4f}")
        if not (mask.min() >= 0 and mask.max() <= 1):
            print(f"Mask {i} has values outside [0, 1] range:")
            print(f"  Min: {mask.min().item():.4f}, Max: {mask.max().item():.4f}")
    print("Data range check completed.")

def check_mask_values(dataset, num_samples=10):
    """
    Check the unique values in masks and their distribution.
    """
    for i in range(num_samples):
        _, mask, _ = dataset[i]
        unique_values = torch.unique(mask)
        print(f"Mask {i} unique values: {unique_values}")

        # Print histogram of mask values
        hist = torch.histc(mask, bins=10, min=0, max=1)
        print(f"Mask {i} value distribution:")
        for j, count in enumerate(hist):
            print(f"  [{j / 10:.1f}-{(j + 1) / 10:.1f}]: {count}")
        print()

    print("Mask value check completed.")


def check_data_consistency(train_dataset, val_dataset, test_dataset):
    """
    Check for data leakage between train, validation, and test sets.
    """
    train_images = set(train_dataset.data['image file path'])
    val_images = set(val_dataset.data['image file path'])
    test_images = set(test_dataset.data['image file path'])

    train_val_overlap = train_images.intersection(val_images)
    train_test_overlap = train_images.intersection(test_images)
    val_test_overlap = val_images.intersection(test_images)

    if len(train_val_overlap) > 0:
        print(f"Data leakage between train and validation sets: {len(train_val_overlap)} images")
        print("Sample overlapping images:")
        for img in list(train_val_overlap)[:5]:
            print(img)

    if len(train_test_overlap) > 0:
        print(f"Data leakage between train and test sets: {len(train_test_overlap)} images")

    if len(val_test_overlap) > 0:
        print(f"Data leakage between validation and test sets: {len(val_test_overlap)} images")

    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("No data leakage detected between datasets.")
    else:
        print("Data leakage detected. Please fix the dataset split.")

def check_label_consistency(dataset):
    """
    Check if labels are consistent with the content of the images and masks.
    """
    for i in range(len(dataset)):
        image, mask, label = dataset[i]
        if label == 0:  # Assuming 0 is for benign
            assert mask.sum() < 0.3 * mask.numel(), f"Benign sample {i} has significant mask area"
        else:  # Malignant
            assert mask.sum() > 0, f"Malignant sample {i} has empty mask"
    print("Labels are consistent with image and mask content.")

# Additional checks
check_data_range(train_dataset)
check_mask_values(train_dataset)
check_data_consistency(train_dataset, val_dataset, test_dataset)
check_label_consistency(train_dataset)

"""def overlay_mask(image, mask, color=[1, 0, 0, 0.5]):  # Red with 50% opacity
    mask = mask.squeeze().numpy()  # Ensure mask is 2D
    colored_mask = np.zeros((*mask.shape, 4))
    colored_mask[mask > 0] = color

    plt.imshow(image.squeeze(), cmap='gray')
    plt.imshow(colored_mask)
    plt.axis('off')


def visualize_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        image, mask, label = dataset[idx]

        axes[i, 0].imshow(image.squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Image (Label: {label})")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask.squeeze(), cmap='gray')
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis('off')

        overlay_mask(image, mask, ax=axes[i, 2])
        axes[i, 2].set_title("Image with Mask")

    plt.tight_layout()
    plt.show()

visualize_samples(train_dataset)"""

import torchvision.transforms.functional as TF

def visualize_augmented_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        original_image, _, label = dataset[idx]

        # Convert tensor to PIL Image for augmentation
        pil_image = TF.to_pil_image(original_image)

        # Apply augmentations
        augmented_image =image_transform(pil_image)

        axes[i, 0].imshow(original_image.squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Original (Label: {label})")
        axes[i, 1].imshow(augmented_image.squeeze(), cmap='gray')
        axes[i, 1].set_title("Augmented")
    plt.tight_layout()
    plt.show()

visualize_augmented_samples(train_dataset)

def verify_data_loading(dataset, num_samples=5):
    """
    Verify that images, masks, and labels are correctly linked by visualizing random samples.
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask, label = dataset[idx]

        axes[i, 0].imshow(image.squeeze(), cmap='gray')
        axes[i, 0].set_title(f"Image (Label: {label})")
        axes[i, 1].imshow(mask.squeeze(), cmap='gray')
        axes[i, 1].set_title("Mask")
        axes[i, 2].imshow(image.squeeze() * mask.squeeze(), cmap='gray')
        axes[i, 2].set_title("Image with Mask")

    plt.tight_layout()
    plt.show()

def verify_label_distribution(dataset):
    """
    Verify the distribution of labels in the dataset.
    """
    labels = [dataset[i][2] for i in range(len(dataset))]
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

def verify_image_mask_correspondence(dataset, num_samples=5):
    """
    Verify that images and masks correspond to each other.
    """
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask, label = dataset[idx]

        # Check if mask is non-zero where the image has content
        image_content = (image.squeeze() > 0.1).float()
        mask_content = (mask.squeeze() > 0.1).float()

        overlap = (image_content * mask_content).sum() / mask_content.sum()

        print(f"Sample {i + 1}: Overlap between image content and mask: {overlap:.2f}")

# Add these verification steps after creating your datasets
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

# Modify the DataLoader to handle the graph data
def collate_fn(batch):
    images, masks, labels = zip(*batch)
    images = torch.stack(images)
    masks = torch.stack(masks)
    labels = torch.tensor(labels)
    return images, masks, labels

from torch.utils.data import WeightedRandomSampler

def create_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

train_sampler = create_weighted_sampler(train_dataset.labels)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

def verify_batch(dataloader):
    images, masks, labels = next(iter(dataloader))
    print(f"Batch shape: Images {images.shape}, Masks {masks.shape}, Labels {labels.shape}")
    print(f"Labels in batch: {labels}")
    print(f"Image dtype: {images.dtype}, Mask dtype: {masks.dtype}, Label dtype: {labels.dtype}")

verify_batch(train_loader)

def verify_labels(dataset, num_samples=10):
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        _, _, label = dataset[idx]
        print(f"Sample {i+1}: Label = {label}")

verify_labels(train_dataset)

# Call dataloader_visualisations function
visualisation_choice_2 = int(input("Do you want to visualise the dataloader? (1 for yes, 0 for no): "))
if visualisation_choice_2 == 1:
    dataloader_visualisations(train_dataset, test_dataset, train_loader, test_loader)
else:
    print("Dataloader visualisation skipped.")

# Initialize the ViT-GNN hybrid model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#model = ViTGNNHybrid(num_classes=2, dropout_rate=0.3).to(device)
model = SimpleCNN(num_classes=2).to(device)
# Calculate class weights
"""class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.labels.numpy()), y=train_dataset.labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)"""

# Define loss function with class weights
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        return F_loss.mean()

criterion = FocalLoss()

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

num_epochs = 5

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Initialize lists to store metrics for plotting
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

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
num_epochs = 5
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        with torch.no_grad():  # Add this line
            preds = F.softmax(outputs, dim=1)[:, 1]  # Get probabilities for the positive class
            train_preds.extend(preds.cpu().detach().numpy())  # Add detach() here
        train_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader)
    train_preds = np.array(train_preds)
    train_labels = np.array(train_labels)
    train_acc = accuracy_score(train_labels, (train_preds > 0.5).astype(int))

    # Validation
    val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_cm, _, _ = evaluate(model, val_loader, device)

    # Store metrics for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Print metrics
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
    print(
        f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')
    print("Validation Confusion Matrix:")
    print(val_cm)
    print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

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
plt.close()