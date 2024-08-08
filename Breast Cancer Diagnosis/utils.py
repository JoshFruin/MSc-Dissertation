import random
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

class FocalLoss(nn.Module):
    """
    Implements Focal Loss, a loss function that addresses class imbalance problems.

    Focal Loss applies a modulating term to the cross entropy loss, in order to focus
    on hard, misclassified examples.

    Attributes:
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float): Focusing parameter for modulating factor (1-p). Default: 2.
        reduction (str): Specifies the reduction to apply to the output. Default: 'mean'.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Args:
            inputs (torch.Tensor): Predicted class scores, typically of shape (N, C) where N is the batch size and C is the number of classes.
            targets (torch.Tensor): Ground truth class indices, typically of shape (N,) where N is the batch size.

        Returns:
            torch.Tensor: Computed focal loss.
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedFocalLoss(nn.Module):
    """
    Implements Weighted Focal Loss, which combines class weighting with Focal Loss.

    This loss function is particularly useful for highly imbalanced datasets, as it
    considers both the class imbalance and the difficulty of classifying each sample.

    Attributes:
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float): Focusing parameter for modulating factor (1-p). Default: 2.
        reduction (str): Specifies the reduction to apply to the output. Default: 'mean'.
        class_weights (torch.Tensor): Computed weights for each class based on their frequency in the dataset.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = None

    def update_class_weights(self, targets):
        """
        Compute and update class weights based on the frequency of each class in the targets.

        Args:
            targets (torch.Tensor): Ground truth class indices.
        """
        if self.class_weights is None:
            class_counts = torch.bincount(targets)
            total_samples = class_counts.sum()
            class_weights = total_samples / (len(class_counts) * class_counts.float())
            self.class_weights = class_weights.to(targets.device)

    def forward(self, inputs, targets):
        """
        Compute the weighted focal loss.

        Args:
            inputs (torch.Tensor): Predicted class scores, typically of shape (N, C) where N is the batch size and C is the number of classes.
            targets (torch.Tensor): Ground truth class indices, typically of shape (N,) where N is the batch size.

        Returns:
            torch.Tensor: Computed weighted focal loss.
        """
        self.update_class_weights(targets)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AlignedTransform:
    """
    Perform safe augmentations for mammogram images and align mask and image transforms.

    This class implements various image augmentation techniques that are suitable for
    medical imaging, particularly mammograms. It ensures that the same transformations
    are applied to both the image and its corresponding mask.

    Attributes:
        size (tuple): Target size for resizing. Default: (224, 224)
        flip_prob (float): Probability of horizontal flip. Default: 0.5
        rotate_prob (float): Probability of rotation. Default: 0.5
        max_rotation (float): Maximum rotation angle in degrees. Default: 10
        brightness_range (tuple): Range for random brightness adjustment. Default: (0.9, 1.1)
        contrast_range (tuple): Range for random contrast adjustment. Default: (0.9, 1.1)
        crop_prob (float): Probability of random crop. Default: 0.3
        crop_scale (tuple): Range of size of the origin size cropped. Default: (0.8, 1.0)
        crop_ratio (tuple): Range of aspect ratio of the origin aspect ratio cropped. Default: (0.9, 1.1)
        noise_prob (float): Probability of adding random noise. Default: 0.3
        noise_factor (float): Strength of the random noise. Default: 0.05
    """
    def __init__(self, size=(224, 224), flip_prob=0.5, rotate_prob=0.5, max_rotation=10,
                 brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1),
                 crop_prob=0.3, crop_scale=(0.8, 1.0), crop_ratio=(0.9, 1.1),
                 noise_prob=0.3, noise_factor=0.05):
        self.size = size
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_rotation = max_rotation
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.crop_prob = crop_prob
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.noise_prob = noise_prob
        self.noise_factor = noise_factor
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self.elastic_transform = transforms.ElasticTransform(alpha=50.0, sigma=5.0)

    def __call__(self, image, mask):
        """
        Apply the configured transformations to both the image and mask.

        Args:
            image (PIL.Image or torch.Tensor): Input image to be transformed.
            mask (PIL.Image or torch.Tensor): Corresponding mask to be tr
            ansformed.

        Returns:
            tuple: Transformed image and mask as torch.Tensor objects.
        """
        # Convert to PIL Image if tensor
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        if isinstance(mask, torch.Tensor):
            mask = TF.to_pil_image(mask)

        # Ensure image is grayscale
        image = image.convert("L")

        # Random crop with adjusted parameters
        if random.random() < self.crop_prob:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=self.crop_scale, ratio=self.crop_ratio)
            image = TF.resized_crop(image, i, j, h, w, self.size)
            mask = TF.resized_crop(mask, i, j, h, w, self.size)
        else:
            image = TF.resize(image, self.size)
            mask = TF.resize(mask, self.size)

        # Random horizontal flip (mirroring)
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random rotation (reduced max angle)
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)

        # Random brightness adjustment (more subtle)
        if random.random() < 0.5:
            brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            image = TF.adjust_brightness(image, brightness_factor)

        # Random contrast adjustment (more subtle)
        if random.random() < 0.5:
            contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            image = TF.adjust_contrast(image, contrast_factor)

        # Add random noise (simulating image acquisition variations)
        if random.random() < self.noise_prob:
            image = TF.to_tensor(image)
            noise = torch.randn(image.size()) * self.noise_factor
            image = torch.clamp(image + noise, 0, 1)
            image = TF.to_pil_image(image)

        # Add elastic deformation
        if random.random() < 0.3:
            image = self.elastic_transform(image)
            mask = self.elastic_transform(mask)

        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Ensure pixel values are in [0, 1] range
        image = torch.clamp(image, 0, 1)

        # Normalize (only for image)
        image = self.normalize(image)

        # Repeat the grayscale channel 3 times to create a 3-channel image
        image = image.repeat(3, 1, 1)

        return image, mask

def balanced_sampling(X, y, target_ratio):
    """
    Perform both oversampling of the minority class and undersampling of the majority class
    to balance the dataset.

    This function aims to address class imbalance by adjusting the number of samples in each class.
    It first oversamples the minority class and then undersamples the majority class to achieve
    the desired ratio between classes.

    Args:
        X (array-like): Feature matrix
        y (array-like): Target vector
        target_ratio (float): Desired ratio of minority to majority class.
                              Should be between 0.0 and 1.0.

    Returns:
        tuple: X_resampled (array-like): The feature matrix after resampling
               y_resampled (array-like): The target vector after resampling
    """
    # Count the occurrences of each class
    class_counts = Counter(y)

    # Identify majority and minority classes
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    # Calculate the target number of samples for each class
    target_minority_count = int(class_counts[majority_class] * target_ratio)

    # Ensure the target count for minority class is valid
    target_minority_count = max(target_minority_count, class_counts[minority_class])

    # Perform oversampling of the minority class
    over = RandomOverSampler(sampling_strategy={minority_class: target_minority_count})
    X_oversampled, y_oversampled = over.fit_resample(X, y)

    # Recount the occurrences of each class after oversampling
    new_class_counts = Counter(y_oversampled)

    # Calculate the target number of samples for the majority class after oversampling
    target_majority_count = int(new_class_counts[minority_class] / target_ratio)

    # Ensure the target count for majority class is valid
    target_majority_count = min(target_majority_count, class_counts[majority_class])

    # Perform undersampling of the majority class
    under = RandomUnderSampler(sampling_strategy={majority_class: target_majority_count})
    X_resampled, y_resampled = under.fit_resample(X_oversampled, y_oversampled)

    return X_resampled, y_resampled