import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

def fix_image_paths(df, image_dir):
    return df.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))

def create_image_dict(image_series):
    image_dict = {}
    for dicom in image_series:
        key = dicom.split("/")[-2]
        image_dict[key] = dicom
    return image_dict

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

        if dataset.iloc[i, 11] in full_mammogram_dict.values() and dataset.iloc[i, 13] in roi_mask_dict.values():
            valid_entries.append(i)

    return dataset.iloc[valid_entries]

class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        self.labels = torch.tensor(self.data['pathology'].map({'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}).fillna(0).astype(int).values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 11]
        mask_name = self.data.iloc[idx, 13]
        if pd.isnull(img_name):
            return None

        if pd.isnull(mask_name):
            mask = None
        else:
            mask = Image.open(mask_name).convert("L")

        image = Image.open(img_name).convert("L")

        if self.transform is not None:
            image = self.transform(image)
            if mask is not None:
                mask = self.transform(mask)

        label = self.labels[idx]
        return image, mask, label
