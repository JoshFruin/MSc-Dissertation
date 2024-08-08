import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random
import os
import torchvision.transforms.functional as TF


def verify_data_linkage(original_csv_path, processed_df, full_mammogram_dict, roi_mask_dict,
                        num_samples=5):
    print(f"Verifying {num_samples} random samples...")

    # Load original CSV
    original_df = pd.read_csv(original_csv_path)

    # Determine the number of samples to verify
    num_samples = min(num_samples, len(processed_df))

    if num_samples == 0:
        print("No samples available for verification.")
        return

    # Get unique indices to sample
    sample_indices = random.sample(range(len(processed_df)), num_samples)

    for i, idx in enumerate(sample_indices):
        # Select a row from the processed dataframe
        processed_row = processed_df.iloc[idx]

        # Find the corresponding row in the original CSV
        matching_rows = original_df[original_df['patient_id'] == processed_row['patient_id']]

        if matching_rows.empty:
            print(f"Error: No matching patient_id found in original CSV for patient_id {processed_row['patient_id']}")
            continue

        original_row = matching_rows.iloc[0]

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
        print(f"  Patient ID: {processed_row['patient_id']}")
        print(f"  Original Image Path: {original_img_path}")
        print(f"  Processed Image Path: {processed_img_path}")
        print(f"  Original Mask Path: {original_mask_path}")
        print(f"  Processed Mask Path: {processed_mask_path}")
        print(f"  Label: {label}")
        print("\n")

    print("Data linkage verification completed.")

def verify_dataset_integrity(dataset, num_samples=5):
    """
    Verify the integrity of the dataset by checking random samples.

    This function performs the following checks on random samples:
    1. Verifies image and mask shapes
    2. Ensures labels are within the expected range
    3. Visualizes the image, mask, and overlay for manual inspection

    Args:
    dataset: The dataset to verify. Should support indexing and return (image, mask, label) tuples.
    num_samples (int): Number of random samples to verify. Default is 5.

    Returns:
    None. Prints verification results and displays images for visual inspection.
    """
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

def check_mask_values(dataset, num_samples=10):
    """
    Check the unique values in masks and their distribution.

    This function examines random samples from the dataset and:
    1. Prints the unique values found in each mask
    2. Displays a histogram of mask values

    This helps identify any unexpected values or distributions in the masks.

    Args:
    dataset: The dataset to check. Should support indexing and return (image, mask, label) tuples.
    num_samples (int): Number of random samples to check. Default is 10.

    Returns:
    None. Prints the results of the mask value checks.
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

def check_data_consistency(train_dataset, val_dataset, test_dataset, image_path_column='image_file_path'):
    """
    Check for data leakage between train, validation, and test sets.

    This function ensures that there is no overlap between the different dataset splits.
    It performs the following checks:
    1. Identifies any images that appear in multiple datasets
    2. Reports the number of overlapping images, if any
    3. Prints sample overlapping image paths for further investigation

    Args:
    train_dataset: The training dataset. Should have a 'data' attribute with image file path column.
    val_dataset: The validation dataset. Should have a 'data' attribute with image file path column.
    test_dataset: The test dataset. Should have a 'data' attribute with image file path column.
    image_path_column: The name of the column containing image file paths. Default is 'image_file_path'.

    Returns:
    None. Prints the results of the consistency check.
    """
    try:
        train_images = set(train_dataset.data[image_path_column])
        val_images = set(val_dataset.data[image_path_column])
        test_images = set(test_dataset.data[image_path_column])

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

    except AttributeError:
        print("Error: Dataset objects do not have a 'data' attribute. Please check your dataset implementation.")
    except KeyError:
        print(f"Error: '{image_path_column}' column not found in the dataset. Please check the column name.")
    except Exception as e:
        print(f"An unexpected error occurred during data consistency check: {str(e)}")

def check_label_consistency(dataset):
    """
    Check if labels are consistent with the content of the images and masks.

    This function verifies that:
    1. Benign samples (label 0) have a relatively small mask area
    2. Malignant samples (label 1) have a non-empty mask

    Args:
    dataset: The dataset to check. Should support indexing and return (image, mask, label) tuples.

    Returns:
    None. Prints the result of the consistency check or raises an AssertionError if inconsistencies are found.
    """
    for i in range(len(dataset)):
        image, mask, label = dataset[i]
        if label == 0:  # Assuming 0 is for benign
            assert mask.sum() < 0.3 * mask.numel(), f"Benign sample {i} has significant mask area"
        else:  # Malignant
            assert mask.sum() > 0, f"Malignant sample {i} has empty mask"
    print("Labels are consistent with image and mask content.")

def visualize_augmented_samples(dataset, image_transform, num_samples=5):
    """
    Visualize original and augmented samples from the dataset.

    This function:
    1. Selects random samples from the dataset
    2. Applies the specified image transformations
    3. Displays the original and augmented images side by side

    Args:
    dataset: The dataset to sample from. Should support indexing and return (image, mask, label) tuples.
    image_transform: The transformation to apply to the images.
    num_samples (int): Number of samples to visualize. Default is 5.

    Returns:
    None. Displays the original and augmented images.
    """
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

def verify_data_loading(dataset, num_samples=5):
    """
    Verify that images, masks, and labels are correctly linked by visualizing random samples.

    This function:
    1. Selects random samples from the dataset
    2. Displays the image, mask, and image with mask overlay
    3. Shows the label for each sample

    Args:
    dataset: The dataset to verify. Should support indexing and return (image, mask, label) tuples.
    num_samples (int): Number of random samples to visualize. Default is 5.

    Returns:
    None. Displays the visualizations for manual inspection.
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

    This function:
    1. Extracts labels from all samples in the dataset
    2. Computes the frequency of each unique label
    3. Visualizes the label distribution as a bar plot

    Args:
    dataset: The dataset to verify. Should support indexing and return (image, mask, label) tuples.

    Returns:
    None. Displays a bar plot of the label distribution.
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

    This function:
    1. Selects random samples from the dataset
    2. Computes the overlap between image content and mask
    3. Prints the overlap percentage for each sample

    A high overlap indicates good correspondence between the image and its mask.

    Args:
    dataset: The dataset to verify. Should support indexing and return (image, mask, label) tuples.
    num_samples (int): Number of random samples to check. Default is 5.

    Returns:
    None. Prints the overlap percentages for the selected samples.
    """
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask, label = dataset[idx]

        # Check if mask is non-zero where the image has content
        image_content = (image.squeeze() > 0.1).float()
        mask_content = (mask.squeeze() > 0.1).float()

        overlap = (image_content * mask_content).sum() / mask_content.sum()

        print(f"Sample {i + 1}: Overlap between image content and mask: {overlap:.2f}")

def verify_batch(dataloader):
    """
    Verify the structure and content of a batch from the dataloader.

    This function:
    1. Retrieves a single batch from the dataloader
    2. Prints the shapes of images, masks, and labels in the batch
    3. Displays the labels present in the batch
    4. Shows the data types of images, masks, and labels

    This helps ensure that the dataloader is correctly formatting and returning batches.

    Args:
    dataloader: A PyTorch DataLoader object to verify.

    Returns:
    None. Prints information about the batch for manual verification.
    """
    images, masks, labels = next(iter(dataloader))
    print(f"Batch shape: Images {images.shape}, Masks {masks.shape}, Labels {labels.shape}")
    print(f"Labels in batch: {labels}")
    print(f"Image dtype: {images.dtype}, Mask dtype: {masks.dtype}, Label dtype: {labels.dtype}")

def verify_labels(dataset, num_samples=10):
    """
    Verify the labels of random samples from the dataset.

    This function:
    1. Selects random samples from the dataset
    2. Prints the label for each selected sample

    This helps in quick manual verification of label correctness and distribution.

    Args:
    dataset: The dataset to verify. Should support indexing and return (image, mask, label) tuples.
    num_samples (int): Number of random samples to check. Default is 10.

    Returns:
    None. Prints the labels of the selected samples for manual verification.
    """
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        _, _, label = dataset[idx]
        print(f"Sample {i+1}: Label = {label}")

def check_and_remove_data_leakage(train_df, val_df, image_path_column='image_file_path'):
    """
    Check for data leakage between train and validation sets and reassign overlapping images alternatively.

    Args:
    train_df (pd.DataFrame): The training DataFrame.
    val_df (pd.DataFrame): The validation DataFrame.
    image_path_column (str): The name of the column containing image file paths. Default is 'image_file_path'.

    Returns:
    pd.DataFrame, pd.DataFrame: Updated DataFrames with reassigned overlapping images.
    """
    try:
        train_images = set(train_df[image_path_column])
        val_images = set(val_df[image_path_column])

        train_val_overlap = train_images.intersection(val_images)

        if len(train_val_overlap) > 0:
            print(f"Data leakage between train and validation sets: {len(train_val_overlap)} images")
            print("Reassigning overlapping images...")

            overlap_list = list(train_val_overlap)
            for i, img in enumerate(overlap_list):
                if i % 2 == 0:
                    # Remove from train and add to val
                    train_df = train_df[train_df[image_path_column] != img]
                    val_df = pd.concat([val_df, train_df[train_df[image_path_column] == img]])
                else:
                    # Remove from val and add to train
                    val_df = val_df[val_df[image_path_column] != img]
                    train_df = pd.concat([train_df, val_df[val_df[image_path_column] == img]])

            # Reset indexes
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        else:
            print("No data leakage detected between train and validation sets.")

        return train_df, val_df

    except KeyError:
        print(f"Error: '{image_path_column}' column not found in the dataset. Please check the column name.")
        return train_df, val_df
    except Exception as e:
        print(f"An unexpected error occurred during data consistency check: {str(e)}")
        return train_df, val_df