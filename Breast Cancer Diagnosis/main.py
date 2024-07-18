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
from torchvision import transforms
from torch.utils.data import Dataset #, DataLoader
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bars
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef

# Imports within the project
from data_visualisations import visualise_data, dataloader_visualisations
from models import SimpleCNN, ResNetClassifier, UNet, VGGClassifier, HybridModel, GNNModel

# Suppress all warnings globally
warnings.filterwarnings("ignore")

#Set up device agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

# Optional: Reset the index of the combined DataFrames
mam_train_data.reset_index(drop=True, inplace=True)
mam_test_data.reset_index(drop=True, inplace=True)

# Data Visualization
"""visualisation_choice = int(input("Do you want to visualise the data? (1 for yes, 0 for no): "))
if visualisation_choice == 1:
    visualise_data(mass_train, calc_train)
else:
    print("Data visualisation skipped.")"""

# Update BreastCancerDataset constructor to print unique values in 'pathology' column before mapping
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

        # Print unique values in the "pathology" column before mapping
        print("Unique Labels Before Mapping:", self.data['pathology'].unique())

        # Map label values to integers
        self.labels = torch.tensor(self.data['pathology'].map({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}).fillna(0).astype(int).values, dtype=torch.long)

        # Print unique values in the "pathology" column after mapping
        print("Unique Labels After Mapping:", self.data['pathology'].unique())

        # Calculate the number of unique classes
        n_classes = len(self.data['pathology'].unique())

        # Debugging: Ensure labels are within valid range
        print("Number of Classes:", n_classes)
        assert torch.all(self.labels >= 0) and torch.all(self.labels < n_classes), "Labels are out of range"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 11]  # Full mammogram image
        mask_name = self.data.iloc[idx, 13]  # ROI mask

        # If image is missing, return None
        if pd.isnull(img_name):
            return None

        # Handle missing mask values
        if pd.isnull(mask_name):
            mask = None
        else:
            mask = Image.open(mask_name).convert("L")  # Ensure mask is grayscale

        image = Image.open(img_name).convert("L")  # Ensure image is grayscale

        # Convert grayscale image to 3 channels by replicating the single channel
        image = image.convert("RGB")

        # Apply transforms separately for image and mask
        if self.transform is not None:
            image = self.transform(image)
            if mask is not None:
                mask = self.transform(mask)

        # Convert mask to binary (0, 1)
        if mask is not None:
            mask = torch.where(mask > 0.5, torch.tensor(1), torch.tensor(0))

        # Extract label from dataframe
        label = self.labels[idx]

        return image.to(device), mask.to(device) if mask is not None else mask, label.to(device)

import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import ToUndirected
class BreastCancerGraphDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

        self.labels = torch.tensor(self.data['pathology'].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 11]  # Full mammogram image
        image = Image.open(img_name).convert("L")  # Ensure image is grayscale

        if self.transform is not None:
            image = self.transform(image)

        # Convert image to graph
        data = self.image_to_graph(image)
        data.y = self.labels[idx]

        return data

    def image_to_graph(self, image):
        # Assuming image is a 2D tensor after transform
        image = image.squeeze(0)  # Remove the channel dimension

        # Create a grid graph from the image pixels
        edge_index = torch_geometric.utils.grid(
            height=image.size(1),
            width=image.size(0)
        )
        x = image.flatten().unsqueeze(1)  # Node features

        data = Data(x=x, edge_index=edge_index)
        return data


class BreastCancerGraphDataset(Dataset):
    def __init__(self, dataframe, transform=None, num_nodes=100):
        self.data = dataframe
        self.transform = transform
        self.num_nodes = num_nodes
        self.labels = torch.tensor(
            self.data['pathology'].map({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}).fillna(0).astype(
                int).values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 11]  # Full mammogram image

        if pd.isnull(img_name):
            return None

        image = Image.open(img_name).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Convert image to graph
        node_features = image.view(-1, 3)[:self.num_nodes]

        # If we have fewer pixels than num_nodes, pad with zeros
        if node_features.size(0) < self.num_nodes:
            padding = torch.zeros(self.num_nodes - node_features.size(0), 3)
            node_features = torch.cat([node_features, padding], dim=0)

        edge_index = self._create_edge_index(self.num_nodes)

        label = self.labels[idx]

        return Data(x=node_features, edge_index=edge_index, y=label)

    def _create_edge_index(self, num_nodes):
        # Create a simple fully connected graph
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        return edge_index  # No need to concatenate with flipped version
# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalization for grayscale images
])

# Use a smaller subset of the dataset for initial experiments
sample_size = 150  # Can go up to 300 from testing but not 400
mam_train_data_sample = mam_train_data.sample(n=sample_size, random_state=42)
mam_test_data_sample = mam_test_data.sample(n=sample_size, random_state=42)

# Ensure the 'pathology' column in the sample datasets contains only integers
mam_train_data_sample['pathology'] = mam_train_data_sample['pathology'].map(label_mapping).astype(int)
mam_test_data_sample['pathology'] = mam_test_data_sample['pathology'].map(label_mapping).astype(int)

# Check for any potential non-numeric entries
print(mam_train_data_sample['pathology'].unique())
print(mam_test_data_sample['pathology'].unique())

# Initialize datasets and dataloaders
"""train_dataset = BreastCancerDataset(dataframe=mam_train_data, transform=transform)
test_dataset = BreastCancerDataset(dataframe=mam_test_data, transform=transform)"""
# Testing datasets
"""train_dataset = BreastCancerDataset(dataframe=mam_train_data_sample, transform=transform)
test_dataset = BreastCancerDataset(dataframe=mam_test_data_sample, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"""

from torch_geometric.data import DataLoader
num_nodes = 100
# Initialize datasets and dataloaders
train_dataset = BreastCancerGraphDataset(dataframe=mam_train_data, transform=transform, num_nodes=num_nodes)
test_dataset = BreastCancerGraphDataset(dataframe=mam_test_data, transform=transform, num_nodes=num_nodes)

import time
start_time = time.time()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Call dataloader_visualisations function
"""visualisation_choice_2 = int(input("Do you want to visualise the dataloader? (1 for yes, 0 for no): "))
if visualisation_choice_2 == 1:
    dataloader_visualisations(train_dataset, test_dataset, train_loader, test_loader)
else:
    print("Dataloader visualisation skipped.")"""

for i in range(10):
    print(train_dataset[i])

# Model setup
num_node_features = 3  # Grayscale image, so single feature per node - grayscale tho
num_classes = 2
model = GNNModel(num_node_features=num_node_features, num_classes=num_classes).to(device)

class_weights = torch.tensor([1.0, (231/148)]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

num_epochs = 5
best_accuracy = 0.0

print("Training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        batch = batch.to(device)

        # Debug prints
        print(f"Batch size: {batch.num_graphs}")
        print(f"x shape: {batch.x.shape}")
        print(f"edge_index shape: {batch.edge_index.shape}")
        print(f"y shape: {batch.y.shape}")

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    scheduler.step(loss)
# Evaluation on test set
model.eval()
y_true = []
y_pred = []
y_scores = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        _, predicted = torch.max(outputs.data, 1)

        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_scores.extend(outputs[:, 1].cpu().numpy())

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)

# Calculate metrics
accuracy = 100 * (y_true == y_pred).sum() / len(y_true)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
specificity = recall_score(y_true, y_pred, pos_label=0)
auc_roc = roc_auc_score(y_true, y_scores)
auc_pr = average_precision_score(y_true, y_scores)
cm = confusion_matrix(y_true, y_pred)

print(f'Accuracy: {accuracy:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'AUC-ROC: {auc_roc:.2f}')
print(f'AUC-PR: {auc_pr:.2f}')
print(f'Confusion Matrix:\n{cm}')

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


"""# Save the model if it has the best accuracy so far
if accuracy > best_accuracy:
    best_accuracy = accuracy
    torch.save(model.state_dict(), 'best_model.pth')"""