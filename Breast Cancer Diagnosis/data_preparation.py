import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Function to fix paths in the DataFrame
def fix_image_paths(df, image_dir):
    return df.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))

# Helper function to create a dictionary from a Series of paths
def create_image_dict(image_series):
    image_dict = {}
    for dicom in image_series:
        key = dicom.split("/")[-2]  # Assuming the key is the folder name before the image name
        image_dict[key] = dicom
    return image_dict

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


# Function to rename columns
def rename_columns(df):
    return df.rename(columns={
        'left or right breast': 'left_or_right_breast',
        'image view': 'image_view',
        'abnormality id': 'abnormality_id',
        'abnormality type': 'abnormality_type',
        'mass shape': 'mass_shape',
        'mass margins': 'mass_margins',
        'calc type': 'calc_type',
        'calc distribution': 'calc_distribution',
        'image file path': 'image_file_path',
        'cropped image file path': 'cropped_image_file_path',
        'ROI mask file path': 'ROI_mask_file_path'
    })

class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None, categorical_columns=None, all_categorical_columns=None):
        self.data = dataframe
        self.transform = transform
        self.categorical_columns = categorical_columns or ['calc_type', 'calc_distribution', 'left_or_right_breast',
                                                           'abnormality_type', 'mass_shape', 'mass_margins']
        self.all_categorical_columns = all_categorical_columns

        # Filter out rows with missing masks
        self.data = self.data.dropna(subset=['ROI_mask_file_path'])

        # Labels are already preprocessed to 0 and 1
        self.labels = torch.tensor(self.data['pathology'].values, dtype=torch.long)

        # Process numerical features
        self.numerical_features = self.data[['subtlety', 'breast_density']].fillna(0)

        # Process categorical features
        self.categorical_features = pd.get_dummies(self.data[self.categorical_columns], columns=self.categorical_columns, dummy_na=True)

        # Ensure all categorical features are present in the dataframe
        for col in self.all_categorical_columns:
            if col not in self.categorical_features:
                self.categorical_features[col] = 0

        # Reorder columns to match the combined dummies columns
        self.categorical_features = self.categorical_features[self.all_categorical_columns]

        # Ensure all categorical features are float
        self.categorical_features = self.categorical_features.astype(float)

        print(f"Shape of categorical features: {self.categorical_features.shape}")
        print(f"Categorical feature dtypes: {self.categorical_features.dtypes.value_counts()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_file_path']
        mask_name = self.data.iloc[idx]['ROI_mask_file_path']

        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("RGB")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        label = self.labels[idx]

        numerical = torch.tensor(self.numerical_features.iloc[idx].values, dtype=torch.float)
        categorical = torch.tensor(self.categorical_features.iloc[idx].values, dtype=torch.float)

        return image, mask, numerical, categorical, label

    def get_num_features(self):
        num_numerical_features = self.numerical_features.shape[1]
        num_categorical_features = self.categorical_features.shape[1]
        return num_numerical_features, num_categorical_features


    @staticmethod
    def get_feature_dimensions(train_df, val_df, test_df, categorical_columns=None):
        categorical_columns = categorical_columns or ['calc_type', 'calc_distribution', 'left_or_right_breast',
                                                      'abnormality_type', 'mass_shape', 'mass_margins']

        # Combine train, validation, and test data for categorical encoding
        combined_df = pd.concat([train_df[categorical_columns], val_df[categorical_columns], test_df[categorical_columns]], axis=0)

        # Get dummies for all possible categories
        all_categories = pd.get_dummies(combined_df, columns=categorical_columns, dummy_na=True)

        num_numerical_features = 2  # subtlety and breast_density
        num_categorical_features = all_categories.shape[1]

        print(f"Number of categorical features after combining train, validation, and test: {num_categorical_features}")

        return num_numerical_features, num_categorical_features, all_categories.columns.tolist()

    def process_breast_density(self):
        if 'breast_density' in self.data.columns:
            self.breast_density = pd.to_numeric(self.data['breast_density'], errors='coerce').fillna(0).astype(int)
        else:
            self.breast_density = pd.Series(np.zeros(len(self.data)), index=self.data.index)

    def process_calc_type(self):
        if 'calc_type' in self.data.columns:
            self.calc_type = pd.get_dummies(self.data['calc_type'], prefix='calc_type')
        else:
            self.calc_type = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                          columns=['calc_type_N/A'])

    def process_calc_distribution(self):
        if 'calc_distribution' in self.data.columns:
            self.calc_distribution = pd.get_dummies(self.data['calc_distribution'], prefix='calc_dist')
        else:
            self.calc_distribution = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                                  columns=['calc_dist_N/A'])

    def process_subtlety(self):
        if 'subtlety' in self.data.columns:
            self.subtlety = pd.to_numeric(self.data['subtlety'], errors='coerce').fillna(0).astype(int)
        else:
            self.subtlety = pd.Series(np.zeros(len(self.data)), index=self.data.index)

    def process_left_or_right(self):
        if 'left_or_right_breast' in self.data.columns:
            self.left_or_right = pd.get_dummies(self.data['left_or_right_breast'], prefix='breast')
        else:
            self.left_or_right = pd.DataFrame(np.zeros((len(self.data), 2)), index=self.data.index,
                                              columns=['breast_LEFT', 'breast_RIGHT'])

    def process_abnormality_type(self):
        if 'abnormality_type' in self.data.columns:
            self.abnormality_type = pd.get_dummies(self.data['abnormality_type'], prefix='abnormality')
        else:
            self.abnormality_type = pd.DataFrame(np.zeros((len(self.data), 2)), index=self.data.index,
                                                 columns=['abnormality_mass', 'abnormality_calcification'])

    def process_mass_shape(self):
        if 'mass_shape' in self.data.columns:
            self.mass_shape = pd.get_dummies(self.data['mass_shape'], prefix='shape')
        else:
            self.mass_shape = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                           columns=['shape_N/A'])

    def process_mass_margins(self):
        if 'mass_margins' in self.data.columns:
            self.mass_margins = pd.get_dummies(self.data['mass_margins'], prefix='margins')
        else:
            self.mass_margins = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                             columns=['margins_N/A'])