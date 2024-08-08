import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

def fix_image_paths(df, image_dir):
    """
    Replace the default image path with the correct image directory.

    Args:
        df (pd.DataFrame): DataFrame containing image paths.
        image_dir (str): The correct image directory path.

    Returns:
        pd.DataFrame: DataFrame with updated image paths.
    """
    return df.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))

def create_image_dict(image_series):
    """
    Create a dictionary mapping folder names to full image paths.

    Args:
        image_series (pd.Series): Series containing image paths.

    Returns:
        dict: Dictionary with folder names as keys and full image paths as values.
    """
    image_dict = {}
    for dicom in image_series:
        key = dicom.split("/")[-2]  # Assuming the key is the folder name before the image name
        image_dict[key] = dicom
    return image_dict

def fix_image_path_mass(dataset, full_mammogram_dict, cropped_dict, roi_mask_dict):
    """
    Fix image paths for mass images and filter out invalid entries.

    Args:
        dataset (pd.DataFrame): DataFrame containing mass image data.
        full_mammogram_dict (dict): Dictionary of full mammogram image paths.
        cropped_dict (dict): Dictionary of cropped image paths.
        roi_mask_dict (dict): Dictionary of ROI mask image paths.

    Returns:
        pd.DataFrame: DataFrame with fixed image paths and only valid entries.
    """
    valid_entries = []
    for i, img in enumerate(dataset.values):
        # Fix full mammogram path
        if len(img) > 11 and isinstance(img[11], str) and '/' in img[11]:
            img_name = img[11].split("/")[-2]
            if img_name in full_mammogram_dict:
                dataset.iloc[i, 11] = full_mammogram_dict[img_name]

        # Fix cropped image path
        if len(img) > 12 and isinstance(img[12], str) and '/' in img[12]:
            img_name = img[12].split("/")[-2]
            if img_name in cropped_dict:
                dataset.iloc[i, 12] = cropped_dict[img_name]

        # Fix ROI mask path
        if len(img) > 13 and isinstance(img[13], str) and '/' in img[13]:
            img_name = img[13].split("/")[-2]
            if img_name in roi_mask_dict:
                dataset.iloc[i, 13] = roi_mask_dict[img_name]

        # Add to valid entries if full mammogram and mask are present
        if dataset.iloc[i, 11] in full_mammogram_dict.values() and dataset.iloc[i, 13] in roi_mask_dict.values():
            valid_entries.append(i)

    return dataset.iloc[valid_entries]

def fix_image_path_calc(dataset, full_mammogram_dict, cropped_dict, roi_mask_dict):
    """
    Fix image paths for calcification images and filter out invalid entries.

    Args:
        dataset (pd.DataFrame): DataFrame containing calcification image data.
        full_mammogram_dict (dict): Dictionary of full mammogram image paths.
        cropped_dict (dict): Dictionary of cropped image paths.
        roi_mask_dict (dict): Dictionary of ROI mask image paths.

    Returns:
        pd.DataFrame: DataFrame with fixed image paths and only valid entries.
    """
    valid_entries = []
    for i, img in enumerate(dataset.values):
        # Fix full mammogram path
        if len(img) > 11 and isinstance(img[11], str) and '/' in img[11]:
            img_name = img[11].split("/")[-2]
            if img_name in full_mammogram_dict:
                dataset.iloc[i, 11] = full_mammogram_dict[img_name]

        # Fix cropped image path
        if len(img) > 12 and isinstance(img[12], str) and '/' in img[12]:
            img_name = img[12].split("/")[-2]
            if img_name in cropped_dict:
                dataset.iloc[i, 12] = cropped_dict[img_name]

        # Fix ROI mask path
        if len(img) > 13 and isinstance(img[13], str) and '/' in img[13]:
            img_name = img[13].split("/")[-2]
            if img_name in roi_mask_dict:
                dataset.iloc[i, 13] = roi_mask_dict[img_name]

        # Add to valid entries if full mammogram and mask are present
        if dataset.iloc[i, 11] in full_mammogram_dict.values() and dataset.iloc[i, 13] in roi_mask_dict.values():
            valid_entries.append(i)

    return dataset.iloc[valid_entries]

def rename_columns(df):
    """
    Rename columns in the DataFrame to more Python-friendly names.

    Args:
        df (pd.DataFrame): Input DataFrame with original column names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
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
    """
    A PyTorch Dataset for breast cancer images and associated metadata.

    This dataset handles both image data and tabular data, including numerical
    and categorical features. It performs necessary preprocessing steps such as
    handling missing values and one-hot encoding of categorical variables.

    Attributes:
        data (pd.DataFrame): The main dataframe containing all data.
        transform (callable, optional): Optional transform to be applied on image data.
        categorical_columns (list): List of categorical column names.
        all_categorical_columns (list): List of all possible categorical column names after one-hot encoding.
        labels (torch.Tensor): Binary labels for each sample.
        numerical_features (pd.DataFrame): Preprocessed numerical features.
        categorical_features (pd.DataFrame): One-hot encoded categorical features.
    """
    def __init__(self, dataframe, transform=None, categorical_columns=None, all_categorical_columns=None):
        """
        Initialize the BreastCancerDataset.

        Args:
            dataframe (pd.DataFrame): The input dataframe containing all data.
            transform (callable, optional): Optional transform to be applied on image data.
            categorical_columns (list, optional): List of categorical column names.
            all_categorical_columns (list, optional): List of all possible categorical column names after one-hot encoding.
        """
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
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing (image, mask, numerical_features, categorical_features, label).
        """
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
        """
        Get the number of numerical and categorical features.

        Returns:
            tuple: A tuple containing (num_numerical_features, num_categorical_features).
        """
        num_numerical_features = self.numerical_features.shape[1]
        num_categorical_features = self.categorical_features.shape[1]
        return num_numerical_features, num_categorical_features

    @staticmethod
    def get_feature_dimensions(train_df, val_df, test_df, categorical_columns=None):
        """
        Get the dimensions of numerical and categorical features across all datasets.

        This static method is used to determine the total number of features and their types
        when combining training, validation, and test datasets.

        Args:
            train_df (pd.DataFrame): Training dataset.
            val_df (pd.DataFrame): Validation dataset.
            test_df (pd.DataFrame): Test dataset.
            categorical_columns (list, optional): List of categorical column names.

        Returns:
            tuple: A tuple containing (num_numerical_features, num_categorical_features, all_categorical_columns).
        """
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
        """Process the breast density feature."""
        if 'breast_density' in self.data.columns:
            self.breast_density = pd.to_numeric(self.data['breast_density'], errors='coerce').fillna(0).astype(int)
        else:
            self.breast_density = pd.Series(np.zeros(len(self.data)), index=self.data.index)

    def process_calc_type(self):
        """Process the calcification type feature."""
        if 'calc_type' in self.data.columns:
            self.calc_type = pd.get_dummies(self.data['calc_type'], prefix='calc_type')
        else:
            self.calc_type = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                          columns=['calc_type_N/A'])

    def process_calc_distribution(self):
        """Process the calcification distribution feature."""
        if 'calc_distribution' in self.data.columns:
            self.calc_distribution = pd.get_dummies(self.data['calc_distribution'], prefix='calc_dist')
        else:
            self.calc_distribution = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                                  columns=['calc_dist_N/A'])

    def process_subtlety(self):
        """Process the subtlety feature."""
        if 'subtlety' in self.data.columns:
            self.subtlety = pd.to_numeric(self.data['subtlety'], errors='coerce').fillna(0).astype(int)
        else:
            self.subtlety = pd.Series(np.zeros(len(self.data)), index=self.data.index)

    def process_left_or_right(self):
        """Process the left or right breast feature."""
        if 'left_or_right_breast' in self.data.columns:
            self.left_or_right = pd.get_dummies(self.data['left_or_right_breast'], prefix='breast')
        else:
            self.left_or_right = pd.DataFrame(np.zeros((len(self.data), 2)), index=self.data.index,
                                              columns=['breast_LEFT', 'breast_RIGHT'])

    def process_abnormality_type(self):
        """Process the abnormality type feature."""
        if 'abnormality_type' in self.data.columns:
            self.abnormality_type = pd.get_dummies(self.data['abnormality_type'], prefix='abnormality')
        else:
            self.abnormality_type = pd.DataFrame(np.zeros((len(self.data), 2)), index=self.data.index,
                                                 columns=['abnormality_mass', 'abnormality_calcification'])

    def process_mass_shape(self):
        """Process the mass shape feature."""
        if 'mass_shape' in self.data.columns:
            self.mass_shape = pd.get_dummies(self.data['mass_shape'], prefix='shape')
        else:
            self.mass_shape = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                           columns=['shape_N/A'])

    def process_mass_margins(self):
        """Process the mass margins feature."""
        if 'mass_margins' in self.data.columns:
            self.mass_margins = pd.get_dummies(self.data['mass_margins'], prefix='margins')
        else:
            self.mass_margins = pd.DataFrame(np.zeros((len(self.data), 1)), index=self.data.index,
                                             columns=['margins_N/A'])

