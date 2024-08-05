import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import torch.nn.functional as F
from PIL import Image
from XAI import GradCAM, apply_colormap, visualize_gradcam

def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs, scheduler):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        device (torch.device): Device to run the training on (CPU or GPU).
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.

    Returns:
        tuple: (epoch_loss, epoch_acc) - The average loss and accuracy for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False) as pbar:
        for inputs, masks, numerical, categorical, labels in train_loader:
            # Move data to the specified device
            inputs, masks, numerical, categorical, labels = inputs.to(device), masks.to(device), numerical.to(device), categorical.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, masks, numerical, categorical)
            loss = criterion(outputs, labels)

            # Backward pass and optimise
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.update(1)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, epoch, num_epochs):
    """
    Validates the model on the validation dataset.

    Args:
        model (torch.nn.Module): The neural network model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the validation on (CPU or GPU).
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.

    Returns:
        tuple: (epoch_loss, epoch_acc) - The average loss and accuracy for this epoch on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False) as pbar:
            for inputs, masks, numerical, categorical, labels in val_loader:
                # Move data to the specified device
                inputs, masks, numerical, categorical, labels = inputs.to(device), masks.to(device), numerical.to(device), categorical.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs, masks, numerical, categorical)
                loss = criterion(outputs, labels)

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.update(1)

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def test_model(model, test_loader, criterion, device):
    """
    Tests the model on the test dataset and computes various metrics.

    Args:
        model (torch.nn.Module): The neural network model to test.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the testing on (CPU or GPU).

    Returns:
        tuple: (accuracy, precision, recall, f1, auc_roc, misclassified_samples, correctly_classified_samples)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    correctly_classified_samples = []
    misclassified_samples = []

    with torch.no_grad():
        for inputs, masks, numerical, categorical, labels in tqdm(test_loader, desc="Testing"):
            # Move data to the specified device
            inputs, masks, numerical, categorical, labels = inputs.to(device), masks.to(device), numerical.to(
                device), categorical.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs, masks, numerical, categorical)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Compute probabilities and predictions
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Store predictions, labels, and probabilities
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(
                probs[:, 1].cpu().numpy())  # Assuming binary classification, use the positive class probability

            # Collect misclassified and correctly classified samples
            misclassified_indices = (preds != labels).nonzero(as_tuple=True)[0]
            correctly_classified_indices = (preds == labels).nonzero(as_tuple=True)[0]

            for idx in misclassified_indices:
                misclassified_samples.append({
                    'input': inputs[idx],
                    'mask': masks[idx],
                    'numerical': numerical[idx],
                    'categorical': categorical[idx],
                    'true_label': labels[idx].item(),
                    'predicted_label': preds[idx].item()
                })

            for idx in correctly_classified_indices[:len(misclassified_indices)]:  # Balance the number of samples
                correctly_classified_samples.append({
                    'input': inputs[idx],
                    'mask': masks[idx],
                    'numerical': numerical[idx],
                    'categorical': categorical[idx],
                    'true_label': labels[idx].item(),
                    'predicted_label': preds[idx].item()
                })

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc_roc = roc_auc_score(all_labels, all_probs)

    # Print results
    print(f"Test Loss: {running_loss / len(test_loader):.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Enhanced ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Calculate True Negative Rate (TNR) and False Negative Rate (FNR)
    tnr = 1 - fpr
    fnr = 1 - tpr

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.plot(fpr, tnr, color='green', lw=2, label='True Negative Rate')
    plt.plot(fpr, fnr, color='red', lw=2, label='False Negative Rate')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Add arrow to indicate direction of threshold change
    plt.annotate('Threshold Decrease', xy=(0.5, 0.5), xytext=(0.6, 0.6),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)  # Calculate AUC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Analyse both correctly classified and misclassified samples with Grad-CAM
    analyze_samples_with_gradcam(model, correctly_classified_samples, misclassified_samples, device)

    return accuracy, precision, recall, f1, auc_roc, misclassified_samples, correctly_classified_samples

def analyze_samples_with_gradcam(model, correctly_classified_samples, misclassified_samples, device, num_samples=3):
    """
    Analyzes correctly classified and misclassified samples using Grad-CAM.

    Args:
        model (torch.nn.Module): The neural network model.
        correctly_classified_samples (list): List of correctly classified samples.
        misclassified_samples (list): List of misclassified samples.
        device (torch.device): Device to run the analysis on (CPU or GPU).
        num_samples (int): Number of samples to analyze from each category.
    """
    gradcam = GradCAM(model, model.efficientnet.features[-1])

    def process_sample(sample, is_correct):
        # Prepare input data
        input_image = sample['input'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        numerical = sample['numerical'].unsqueeze(0).to(device)
        categorical = sample['categorical'].unsqueeze(0).to(device)

        # Generate Grad-CAM heatmap
        heatmap = gradcam.generate_cam((input_image, mask, numerical, categorical))

        # Convert tensor to PIL Image for visualisation
        original_img = Image.fromarray(
            (input_image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        # Apply colormap and overlay
        cam_image = apply_colormap(heatmap, original_img)

        # Process the mask
        mask_np = mask.squeeze().cpu().numpy()
        if mask_np.ndim == 3 and mask_np.shape[0] == 3:
            mask_np = np.mean(mask_np, axis=0)
        mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min())

        # Display results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        ax1.imshow(original_img)
        ax1.set_title(f"{'Correct' if is_correct else 'Incorrect'}\nTrue: {sample['true_label']}, Predicted: {sample['predicted_label']}")
        ax1.axis('off')
        ax2.imshow(cam_image)
        ax2.set_title('Grad-CAM')
        ax2.axis('off')
        ax3.imshow(mask_np, cmap='gray')
        ax3.set_title('Mask')
        ax3.axis('off')
        plt.tight_layout()
        plt.show()

        print(f"Numerical features: {sample['numerical'].cpu().numpy()}")
        print(f"Categorical features: {sample['categorical'].cpu().numpy()}")
        print("\n")

    print("Correctly Classified Samples:")
    for sample in correctly_classified_samples[:num_samples]:
        process_sample(sample, is_correct=True)

    print("Misclassified Samples:")
    for sample in misclassified_samples[:num_samples]:
        process_sample(sample, is_correct=False)

def analyze_misclassifications(misclassified_samples):
    """
    Analyzes misclassified samples by visualizing input images, masks, and features.

    Args:
        misclassified_samples (list): List of misclassified samples.
    """
    for i, sample in enumerate(misclassified_samples[:5]):  # Analyze first 5 misclassified samples
        plt.figure(figsize=(10, 5))

        # Process the input image
        plt.subplot(1, 2, 1)
        input_img = sample['input'].cpu().permute(1, 2, 0).numpy()  # Change shape from (3, 224, 224) to (224, 224, 3)
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())  # Normalize to [0, 1]
        plt.imshow(input_img)
        plt.title(f"True: {sample['true_label']}, Predicted: {sample['predicted_label']}")

        # Process the mask image
        plt.subplot(1, 2, 2)
        mask_img = sample['mask'].cpu().squeeze().numpy()  # Ensure the mask has shape (height, width)
        if mask_img.ndim == 3 and mask_img.shape[0] == 3:
            mask_img = mask_img[0]  # Assume the first channel if the mask is mistakenly in (3, height, width)
        mask_img = (mask_img - mask_img.min()) / (mask_img.max() - mask_img.min())  # Normalize to [0, 1]
        plt.imshow(mask_img, cmap='gray')
        plt.title("Mask")

        # Show the plot and print numerical and categorical features
        plt.show()
        print(f"Numerical features: {sample['numerical'].cpu()}")
        print(f"Categorical features: {sample['categorical'].cpu()}")
        print("\n")

"""def analyze_feature_importance(model):
    # Get the weights of the final fully connected layer
    fc3_weights = model.fc3.weight.data.cpu().numpy()

    # Calculate the absolute sum of weights for each input feature
    feature_importance = np.abs(fc3_weights).sum(axis=0)

    # Normalize the importance scores
    feature_importance = feature_importance / feature_importance.sum()

    # Create labels for each feature group
    image_features = model.fc2.in_features
    mask_features = 32
    numerical_features = 64
    categorical_features = 64

    feature_labels = (
        [f'Image {i+1}' for i in range(image_features)] +
        [f'Mask {i+1}' for i in range(mask_features)] +
        [f'Numerical {i+1}' for i in range(numerical_features)] +
        [f'Categorical {i+1}' for i in range(categorical_features)]
    )

def improved_feature_importance(feature_importance, feature_labels, top_n=20):
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[sorted_idx]
    sorted_labels = [feature_labels[i] for i in sorted_idx]

    # Select top N features
    top_importance = sorted_importance[:top_n]
    top_labels = sorted_labels[:top_n]

    # Create color map
    color_map = {'Image': 'blue', 'Mask': 'green', 'Numerical': 'red', 'Categorical': 'orange'}
    colors = [color_map[label.split()[0]] for label in top_labels]

    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(top_n), top_importance, align='center', color=colors)
    plt.yticks(range(top_n), top_labels)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')

    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in color_map.values()]
    plt.legend(handles, color_map.keys(), loc='lower right')

    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                 ha='left', va='center')

    plt.tight_layout()
    plt.show()"""