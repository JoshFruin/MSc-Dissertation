import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# Suppress all warnings globally
warnings.filterwarnings("ignore")

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

# Apply the function to the mass and calc datasets
fix_image_path_mass(mass_train_data, full_mammogram_dict, cropped_dict, roi_mask_dict)
fix_image_path_mass(mass_test_data, full_mammogram_dict, cropped_dict, roi_mask_dict)

# Check the updated DataFrames (optional, for checking)
print(mass_train_data.head())
print(mass_test_data.head())

def fix_image_path_calc(dataset, full_mammogram_dict, cropped_dict, roi_mask_dict):
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

# Apply the function to the calc datasets
fix_image_path_calc(calc_train_data, full_mammogram_dict, cropped_dict, roi_mask_dict)
fix_image_path_calc(calc_test_data, full_mammogram_dict, cropped_dict, roi_mask_dict)

# Check the updated DataFrames (optional, for checking)
print(calc_train_data.head())
print(calc_test_data.head())

"""##### II. Data Cleaning"""

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

"""##### III. Data Visualization"""

# quantitative summary of features
print(mass_train.describe())
print(calc_train.describe())

# check datasets shape
print(f'Shape of mass_train: {mass_train.shape}')
print(f'Shape of mass_test: {mass_test.shape}')

# check datasets shape
print(f'Shape of calc_train: {calc_train.shape}')
print(f'Shape of calc_test: {calc_test.shape}')

# pathology distributions
value = mass_train['pathology'].value_counts() + calc_train['pathology'].value_counts()
plt.figure(figsize=(8,6))
plt.pie(value, labels=value.index, autopct='%1.1f%%')
plt.title('Breast Cancer Mass Types', fontsize=12)
plt.show()

# Assuming mass_train and calc_train are your DataFrames

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
plt.title('Breast Density vs Pathology\n\n1: fatty || 2: Scattered Fibroglandular Density\n3: Heterogenously Dense || 4: Extremely Dense',
          fontsize=14)
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

print(mass_train.head())
print(calc_train.head())

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

def display_images(column, number):
    """displays images in the dataset"""
    # create figure and axes
    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))

    # Loop through rows and display images
    for index, row in calc_train.head(number_to_visualize).iterrows():
        image_path = row[column]
        print(image_path)
        # Check if the file exists
        if os.path.exists(image_path):
            image = mpimg.imread(image_path)
            # Plot the image
            axes[index].imshow(image, cmap='gray')
            axes[index].set_title(f"{row['pathology']}")
            axes[index].axis('off')
        else:
            print(f"File not found: {image_path}")

    plt.tight_layout()
    plt.show()

print('Calc Training Dataset\n\n')
print('Full Mammograms:\n')
display_images('image_file_path', 5)
print('Cropped Mammograms:\n')
display_images('cropped_image_file_path', 5)
print('ROI Images:\n')
display_images('ROI_mask_file_path', 5)

"""##### V. Data Preprocessing"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Assuming 'mass_train' and 'calc_train' are your DataFrames

# Define a custom dataset class for your data
class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming your dataframe has a column 'image_file_path' containing the file paths to the images
        img_name = self.data.iloc[idx, 11]  # Adjust the column index as needed
        image = Image.open(img_name)

        # Assuming your dataframe has a column 'pathology' containing the labels
        label = self.data.iloc[idx, 10]  # Adjust the column index as needed

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
])

# Define your datasets using the custom dataset class
train_dataset = BreastCancerDataset(dataframe=mass_train, transform=transform)
test_dataset = BreastCancerDataset(dataframe=mass_test, transform=transform)

# Define your dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Assuming you have a CNN model defined

# Define your model
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define your layers here

    def forward(self, x):
        # Define the forward pass of your model here
        return x

# Initialize your model
model = YourModel()

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5
# Assuming you have a CUDA-enabled device available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move your model to the device
model.to(device)

# Train your model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# Evaluate your model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the test images: {(100 * correct / total):.2f}%")
