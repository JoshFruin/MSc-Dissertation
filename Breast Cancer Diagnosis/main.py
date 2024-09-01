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
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange  # Import tqdm for progress bars
from data_visualisations import visualise_data, dataloader_visualisations, visualize_augmentations, verify_augmentations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from PIL import Image
import random
from data_verification import (verify_data_linkage, verify_dataset_integrity,
                               check_mask_values, check_data_consistency,
                               check_label_consistency, visualize_augmented_samples,
                               verify_data_loading, verify_label_distribution,
                               verify_image_mask_correspondence, verify_batch, verify_labels, check_and_remove_data_leakage) #check_data_range,
from models import SimpleCNN, TransferLearningModel, MultimodalModel, B2MultimodalModel,ResMultimodalModel, Res50MultimodalModel, InceptionMultimodalModel
import torchvision.models as models
from collections import Counter
import multiprocessing
# Suppress the specific torchvision warning
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
# Check torch version and if a GPU is available on the device
from torch.utils.data import DataLoader, random_split
from data_preparation import fix_image_paths, create_image_dict, fix_image_path_mass, fix_image_path_calc, rename_columns, BreastCancerDataset
from utils import FocalLoss, WeightedFocalLoss, AlignedTransform, balanced_sampling
from train_test_val import train, validate, test_model, analyze_misclassifications #, analyze_feature_importance
# Suppress all warnings globally
warnings.filterwarnings("ignore")

def get_combined_categorical_columns(train_df, val_df, test_df, categorical_columns):
    """
        Combine categorical columns from train, validation, and test datasets and create dummy variables.

        Args:
        train_df (pd.DataFrame): Training dataset
        val_df (pd.DataFrame): Validation dataset
        test_df (pd.DataFrame): Test dataset
        categorical_columns (list): List of categorical column names

        Returns:
        list: List of all dummy variable column names
        """
    combined_df = pd.concat([train_df[categorical_columns], val_df[categorical_columns], test_df[categorical_columns]], axis=0)
    combined_dummies = pd.get_dummies(combined_df, columns=categorical_columns, dummy_na=True)
    return combined_dummies.columns.tolist()

def main():
    # Set up multiprocessing
    if __name__ == '__main__':
        multiprocessing.set_start_method('spawn', force=True)

    """
    Main function to orchestrate the breast cancer classification pipeline.
    This function handles data loading, preprocessing, model training, and evaluation.
    For the mammograms we're using the CBIS-DDSM dataset from Kaggle: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data
    """

    # Define file paths
    csv_path_meta = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/meta.csv'
    csv_path_dicom = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv'
    mass_train_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv'
    mass_test_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv'
    calc_train_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_train_set.csv'
    calc_test_path = 'C:/Users/jafru/OneDrive - University of Plymouth/MSc Dissertation/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_test_set.csv'

    # Read the CSV files into DataFrames
    df_meta = pd.read_csv(csv_path_meta)
    dicom_data = pd.read_csv(csv_path_dicom)
    mass_train_data = pd.read_csv(mass_train_path)
    mass_test_data = pd.read_csv(mass_test_path)
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

    # Filter and create image dictionaries
    full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
    cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
    roi_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path

    full_mammogram_dict = create_image_dict(full_mammogram_images)
    cropped_dict = create_image_dict(cropped_images)
    roi_mask_dict = create_image_dict(roi_mask_images)

    # Display a sample item from each dictionary
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

    # Check the updated DataFrames
    print(mass_train_data.head())
    print(mass_test_data.head())

    # Apply the function to the calc datasets
    calc_train_data = fix_image_path_calc(calc_train_data, full_mammogram_dict, cropped_dict, roi_mask_dict)
    calc_test_data = fix_image_path_calc(calc_test_data, full_mammogram_dict, cropped_dict, roi_mask_dict)

    # Check the updated DataFrames
    print(calc_train_data.head())
    print(calc_test_data.head())

    # check unique values in pathology column
    mass_train_data.pathology.unique()
    calc_train_data.pathology.unique()
    mass_train_data.info()
    calc_train_data.info()

    # Rename columns for consistency
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

    # Drop rows with missing pathology and convert to int
    mam_train_data = mam_train_data.dropna(subset=['pathology'])
    mam_test_data = mam_test_data.dropna(subset=['pathology'])
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

    X = mam_train_data.drop('pathology', axis=1)
    y = mam_train_data['pathology']

    # Print initial class distribution
    print("Initial class distribution:", Counter(y))

    # Verify data linkage
    print("Verifying data linkage...")
    #verify_data_linkage(mass_train_path, mam_train_data, full_mammogram_dict, roi_mask_dict, num_samples=5)
    #verify_data_linkage(calc_train_path, mam_train_data, full_mammogram_dict, roi_mask_dict, num_samples=5)

    # Balance the dataset
    X_resampled, y_resampled = balanced_sampling(X, y, target_ratio=1.2)

    # Create a new DataFrame with the resampled data
    mam_train_data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    mam_train_data_resampled['pathology'] = y_resampled

    # Print class distribution after resampling
    print("Class distribution after resampling:", Counter(y_resampled))

    # Split the combined training data into train and validation sets
    mam_train_data, mam_val_data = train_test_split(mam_train_data_resampled, test_size=0.2, random_state=42,
                                            stratify=mam_train_data_resampled['pathology'])

    print("Validation set class distribution:", Counter(mam_val_data['pathology']))

    # Check and remove data leakage
    mam_train_data, mam_val_data = check_and_remove_data_leakage(mam_train_data, mam_val_data)

    # Define transforms for data augmentation
    train_transform = AlignedTransform(
        size=(224, 224),
        flip_prob=0.5,
        rotate_prob=0.5,
        max_rotation=3,
        brightness_range=(0.97, 1.03),
        contrast_range=(0.97, 1.03),
        crop_prob=0.2,
        crop_scale=(0.95, 1.0),
        crop_ratio=(0.95, 1.05),
        noise_prob=0.2,
        noise_factor=0.02
    )
    aggressive_train_transform = AlignedTransform(
        size=(224, 224),
        flip_prob=0.5,
        rotate_prob=0.7,  # Increased from 0.5
        max_rotation=5,  # Increased from 3
        brightness_range=(0.95, 1.05),  # Slightly wider range
        contrast_range=(0.95, 1.05),  # Slightly wider range
        crop_prob=0.3,  # Increased from 0.2
        crop_scale=(0.9, 1.0),  # Wider range
        crop_ratio=(0.9, 1.1),  # Wider range
        noise_prob=0.3,  # Increased from 0.2
        noise_factor=0.03  # Increased from 0.02
    )

    val_test_transform = AlignedTransform(
        size=(224, 224),
        flip_prob=0,
        rotate_prob=0,
        max_rotation=0,
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        crop_prob=0,
        crop_scale=(1.0, 1.0),
        crop_ratio=(1.0, 1.0),
        noise_prob=0,
        noise_factor=0
    )

    # Get feature dimensions
    num_numerical_features, num_categorical_features, categorical_feature_columns = BreastCancerDataset.get_feature_dimensions(
        mam_train_data, mam_val_data, mam_test_data)

    # Get combined categorical columns
    categorical_columns = ['calc_type', 'calc_distribution', 'left_or_right_breast', 'abnormality_type', 'mass_shape',
                           'mass_margins']
    all_categorical_columns = get_combined_categorical_columns(mam_train_data, mam_val_data, mam_test_data,
                                                               categorical_columns)

    print(f"Number of categorical features: {len(all_categorical_columns)}")

    # Initialise datasets
    train_dataset = BreastCancerDataset(mam_train_data, transform=train_transform, categorical_columns=categorical_columns, all_categorical_columns=all_categorical_columns)
    val_dataset = BreastCancerDataset(mam_val_data, transform=val_test_transform, categorical_columns=categorical_columns, all_categorical_columns=all_categorical_columns)
    test_dataset = BreastCancerDataset(mam_test_data, transform=val_test_transform, categorical_columns=categorical_columns, all_categorical_columns=all_categorical_columns)

    # Check data consistency
    print("Checking data consistency...")
    check_data_consistency(train_dataset, val_dataset, test_dataset, image_path_column='image_file_path')

    # Visualise augmentations
    visualize_augmentations(train_dataset)
    verify_augmentations(train_dataset)

    # Create dataloaders
    batch_size = 32
    num_workers = min(os.cpu_count(), 6)
    train_loader = DataLoader(train_dataset,  batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Set device (GPU if available, else CPU)
    model = MultimodalModel(num_numerical_features, num_categorical_features, dropout_rate=0.6).to(device)

    # Set up training parameters
    patience = 3 # 5 # Patience for early stopping to prevent overfitting - number means how many epochs to stop on no improvement

    class_counts = Counter(y_resampled)
    total_samples = sum(class_counts.values())
    class_weights = {class_label: total_samples / count for class_label, count in class_counts.items()}
    """Hyperparameters"""
    # Mapping to alpha values, ensuring the minority class gets higher weight
    alpha = class_weights[0]  # The weight for the minority class
    # Balances the dataset
    #criterion = nn.CrossEntropyLoss()
    criterion = WeightedFocalLoss(alpha=alpha, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 30
    # Calculate the total number of training steps
    total_steps = len(train_loader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_val_loss = float('inf')
    no_improve = 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, scheduler)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, num_epochs)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

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

    # Plot training metrics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss vs. Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epochs')
    plt.show()

    # Load best model and evaluate on test set
    print("Loading best model and evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    accuracy, precision, recall, f1, auc_roc, misclassified_samples, correctly_classified_samples = test_model(model, test_loader, criterion, device)

    # Call the analyze_misclassifications function
    analyze_misclassifications(misclassified_samples)

if __name__ == '__main__':
    main()