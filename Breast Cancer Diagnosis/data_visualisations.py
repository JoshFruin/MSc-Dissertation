import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import matplotlib.image as mpimg
import torch

def visualise_data(mass_train, calc_train):
    # Set the color palette for mass_train
    mass_palette = sns.color_palette("viridis", n_colors=len(mass_train['assessment'].unique()))
    sns.countplot(data=mass_train, y='assessment', hue='pathology', palette=mass_palette)
    plt.title('Count Plot for mass_train')
    plt.show()

    # Set the color palette for calc_train
    calc_palette = sns.color_palette("magma", n_colors=len(calc_train['assessment'].unique()))
    sns.countplot(data=calc_train, y='assessment', hue='pathology', palette=calc_palette)
    plt.title('Count Plot for calc_train')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=mass_train, x='subtlety', palette='viridis', hue='subtlety')
    plt.title('Breast Cancer Mass Subtlety', fontsize=12)
    plt.xlabel('Subtlety Grade')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=calc_train, x='subtlety', palette='magma', hue='subtlety')
    plt.title('Breast Cancer Calc Subtlety', fontsize=12)
    plt.xlabel('Subtlety Grade')
    plt.ylabel('Count')
    plt.show()

    # view breast mass shape distribution against pathology
    plt.figure(figsize=(8,6))
    sns.countplot(mass_train, x='mass_shape', hue='pathology')
    plt.title('Mass Shape Distribution by Pathology', fontsize=14)
    plt.xlabel('Mass Shape')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Pathology Count')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.countplot(data=calc_train, y='calc_type', hue='pathology', palette='viridis')
    plt.title('Calcification Type Distribution by Pathology', fontsize=14)
    plt.xlabel('Pathology Count')
    plt.ylabel('Calc Type')
    # Adjust the rotation of the y-axis labels
    plt.yticks(rotation=0, ha='right')
    # Move the legend outside the plot for better visibility
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.show()

    # breast density against pathology
    plt.figure(figsize=(8,6))
    sns.countplot(mass_train, x='breast_density', hue='pathology')
    plt.title('Breast Density vs Pathology\n\n1: fatty || 2: Scattered Fibroglandular Density\n3: Heterogenously Dense || 4: Extremely Dense',fontsize=14)
    plt.xlabel('Density Grades')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    # breast density against pathology
    plt.figure(figsize=(8,6))

    sns.countplot(calc_train, x='breast_density', hue='pathology')
    plt.title('Breast Density vs Pathology\n\n1: fatty || 2: Scattered Fibroglandular Density\n3: Heterogenously Dense || 4: Extremely Dense',
            fontsize=14)
    plt.xlabel('Density Grades')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

    def display_images(column, number):
        """Displays images in the dataset, handling missing files."""
        number_to_visualize = number

        fig = plt.figure(figsize=(15, 5))

        for index, row in mass_train.head(number_to_visualize).iterrows():
            image_path = row[column]
            # print(image_path) # Uncomment this to see printed file paths

            if os.path.exists(image_path):
                image = mpimg.imread(image_path)
                # create axes and display image
                ax = fig.add_subplot(1, number_to_visualize, index + 1)
                ax.imshow(image, cmap='gray')
                ax.set_title(f"{row['pathology']}")
                ax.axis('off')
            else:
                print(f"File not found: {image_path}")  # Log missing files

        plt.tight_layout()
        plt.show()

    print('Mass Training Dataset\n\n')
    print('Full Mammograms:\n')
    display_images('image_file_path', 5)
    print('Cropped Mammograms:\n')
    display_images('cropped_image_file_path', 5)
    print('ROI Images:\n')
    display_images('ROI_mask_file_path', 5)

    print('Calc Training Dataset\n\n')
    print('Full Mammograms:\n')
    display_images('image_file_path', 5)
    print('Cropped Mammograms:\n')
    display_images('cropped_image_file_path', 5)
    print('ROI Images:\n')
    display_images('ROI_mask_file_path', 5)

def show_images(dataset, num_images=5):
    def show_images(dataset, num_images=5):
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
        for i in range(num_images):
            image, mask, label = dataset[i]
            # Display the first channel of the image (assuming it's RGB)
            axes[0, i].imshow(image[0], cmap='gray')
            axes[0, i].set_title(f"Label: {label}")
            axes[0, i].axis('off')

            axes[1, i].imshow(mask.squeeze(), cmap='gray')
            axes[1, i].set_title("Mask")
            axes[1, i].axis('off')
        plt.show()

def visualize_batch(dataloader, num_images=5):
    dataiter = iter(dataloader)
    images, masks, labels = next(dataiter)

    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))

    for i in range(num_images):
        # Display the first channel of the image (assuming it's RGB)
        axes[0, i].imshow(images[i][0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title(f"Label: {labels[i]}")
        axes[0, i].axis('off')

        if masks is not None:  # Check if masks are available
            # Display the mask
            axes[1, i].imshow(masks[i].squeeze().cpu().numpy(), cmap='gray')
            axes[1, i].set_title("Mask")
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def dataloader_visualisations(train_dataset, test_dataset, train_loader, test_loader):
    print("Training Dataset:")
    show_images(train_dataset)

    print("Test Dataset:")
    show_images(test_dataset)

    print("Training Dataloader:")
    visualize_batch(train_loader)

    print("Test Dataloader:")
    visualize_batch(test_loader)

    images, masks, labels = next(iter(train_loader))
    print("Training Dataloader Output:")
    print(f"Images Shape: {images.shape}")
    print(f"Masks Shape: {masks.shape}")
    print(f"Labels Shape: {labels.shape}")

    images, masks, labels = next(iter(test_loader))
    print("Test Dataloader Output:")
    print(f"Images Shape: {images.shape}")
    print(f"Masks Shape: {masks.shape}")
    print(f"Labels Shape: {labels.shape}")

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