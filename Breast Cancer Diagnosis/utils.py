import random
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
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
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = None

    def update_class_weights(self, targets):
        # Compute class weights once and store them
        if self.class_weights is None:
            class_weights = compute_class_weight('balanced', classes=np.unique(targets.cpu()),
                                                 y=targets.cpu().numpy())
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(targets.device)

    def forward(self, inputs, targets):
        self.update_class_weights(targets)

        # Apply class weights to logits
        weighted_inputs = inputs * self.class_weights.unsqueeze(0)

        ce_loss = nn.functional.cross_entropy(weighted_inputs, targets, reduction='none')
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
    """
    def __init__(self, size=(224, 224), flip_prob=0.5, rotate_prob=0.5, max_rotation=10,
                 brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1),
                 crop_prob=0.3, crop_scale=(0.8, 1.0), crop_ratio=(0.75, 1.33)):
        self.size = size
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_rotation = max_rotation
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.crop_prob = crop_prob
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, image, mask):
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

        # Random brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            image = TF.adjust_brightness(image, brightness_factor)

        # Random contrast adjustment
        if random.random() < 0.5:
            contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            image = TF.adjust_contrast(image, contrast_factor)

        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Normalize (only for image)
        image = self.normalize(image)

        # Repeat the grayscale channel 3 times to create a 3-channel image
        image = image.repeat(3, 1, 1)

        return image, mask

def balanced_sampling(X, y, target_ratio):
    """
    Perform both oversampling of the minority class and undersampling of the majority class
    to balance the dataset.

    Parameters:
    X (array-like): Feature matrix
    y (array-like): Target vector
    target_ratio (float): Desired ratio of minority to majority class.
                          Should be between 0.0 and 1.0.

    Returns:
    X_resampled (array-like): The feature matrix after resampling
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