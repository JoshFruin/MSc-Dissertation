import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import random
import os
import torchvision.transforms.functional as TF

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

def visualize_augmented_samples(dataset, image_transform, num_samples=5):
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

def verify_batch(dataloader):
    images, masks, labels = next(iter(dataloader))
    print(f"Batch shape: Images {images.shape}, Masks {masks.shape}, Labels {labels.shape}")
    print(f"Labels in batch: {labels}")
    print(f"Image dtype: {images.dtype}, Mask dtype: {masks.dtype}, Label dtype: {labels.dtype}")

def verify_labels(dataset, num_samples=10):
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        _, _, label = dataset[idx]
        print(f"Sample {i+1}: Label = {label}")