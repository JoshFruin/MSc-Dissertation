import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.functional import relu

class AttentionBlock(nn.Module):
    """
    Implements a simple attention mechanism for feature weighting.

    This block applies self-attention to the input features, allowing the model
    to focus on the most relevant features for the task at hand.
    """
    def __init__(self, in_features):
        """
        Initialize the AttentionBlock.

        Args:
            in_features (int): Number of input features.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Apply attention mechanism to input features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Attention-weighted features of the same shape as input.
        """
        attention_weights = self.attention(x)
        return x * attention_weights

class MultiHeadAttentionBlock(nn.Module):
    """
    Implements a multi-head attention mechanism for enhanced feature interaction.

    This block applies multiple attention heads in parallel, allowing the model
    to capture different types of feature interactions simultaneously.
    """
    def __init__(self, in_features, num_heads):
        """
        Initialize the MultiHeadAttentionBlock.

        Args:
            in_features (int): Number of input features.
            num_heads (int): Number of attention heads.
        """
        super(MultiHeadAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Linear(in_features // 2, in_features),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])
        self.fc = nn.Linear(in_features * num_heads, in_features)

    def forward(self, x):
        """
        Apply multi-head attention mechanism to input features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Multi-head attention-weighted features of shape (batch_size, in_features).
        """
        attentions = [head(x) for head in self.attention_heads]
        x = torch.cat(attentions, dim=-1)
        x = self.fc(x)
        return x

class MultimodalModel(nn.Module):
    """
    A multimodal neural network model that combines image, mask, numerical, and categorical data.

    This model uses EfficientNet-B0 for image feature extraction, applies convolutions to mask data,
    and processes numerical and categorical data through dense layers. It then combines all features
    using multi-head attention before final classification.
    """
    def __init__(self, num_numerical_features, num_categorical_features, dropout_rate=0.5, num_heads=4):
        """
        Initialize the MultimodalModel.

        Args:
            num_numerical_features (int): Number of numerical input features.
            num_categorical_features (int): Number of categorical input features.
            dropout_rate (float): Dropout rate for regularization.
            num_heads (int): Number of attention heads in the multi-head attention block.
        """
        super(MultimodalModel, self).__init__()

        # Image feature extractor (EfficientNet-B0)
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.unfreeze_last_n_layers(self.efficientnet, 20)  # Unfreeze last 20 layers
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        # Mask feature extractor
        self.mask_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mask_bn = nn.BatchNorm2d(32)
        self.mask_layers = nn.Sequential(
            self.mask_conv, self.mask_bn, nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Numerical and categorical feature processors
        self.num_dense = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )
        self.cat_dense = nn.Sequential(
            nn.Linear(num_categorical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )

        # Multi-head attention mechanism
        self.attention = MultiHeadAttentionBlock(num_ftrs + 32 + 64 + 64, num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(num_ftrs + 32 + 64 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def unfreeze_last_n_layers(self, model, n):
        """
        Unfreeze the last n layers of a model for fine-tuning.

        Args:
            model (nn.Module): The model to partially unfreeze.
            n (int): Number of layers to unfreeze from the end.
        """
        params = list(model.parameters())
        for param in params[:-n]:
            param.requires_grad = False
        for param in params[-n:]:
            param.requires_grad = True

    def forward(self, image, mask, numerical, categorical):
        """
        Forward pass of the multimodal model.

        Args:
            image (torch.Tensor): Batch of input images.
            mask (torch.Tensor): Batch of input masks.
            numerical (torch.Tensor): Batch of numerical features.
            categorical (torch.Tensor): Batch of categorical features.

        Returns:
            torch.Tensor: Model output (logits) for each class.
        """
        if categorical.shape[1] != self.cat_dense[0].in_features:
            raise ValueError(
                f"Expected {self.cat_dense[0].in_features} categorical features, but got {categorical.shape[1]}")
        # Process image features
        x_img = self.efficientnet(image)

        # Process mask features
        x_mask = self.mask_layers(mask[:, 0:1, :, :])  # Use only the first channel of the mask
        x_mask = x_mask.view(mask.size(0), -1)

        # Process numerical and categorical feature
        x_num = self.num_dense(numerical)
        x_cat = self.cat_dense(categorical)

        # Combine all features
        combined = torch.cat((x_img, x_mask, x_num, x_cat), dim=1)

        # Apply multi-head attention
        combined = self.attention(combined)

        # Final fully connected layers
        x = F.relu(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class B2MultimodalModel(nn.Module):
    """
    A multimodal neural network model that combines image, mask, numerical, and categorical data.
    This version uses EfficientNet-B2 for image feature extraction.
    """
    def __init__(self, num_numerical_features, num_categorical_features, dropout_rate=0.5, num_heads=4):
        super(B2MultimodalModel, self).__init__()

        # Image feature extractor (EfficientNet-B2)
        self.efficientnet = models.efficientnet_b2(pretrained=True)
        self.unfreeze_last_n_layers(self.efficientnet, 20)  # Unfreeze last 20 layers
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        # Mask feature extractor
        self.mask_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mask_bn = nn.BatchNorm2d(32)
        self.mask_layers = nn.Sequential(
            self.mask_conv, self.mask_bn, nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Numerical and categorical feature processors
        self.num_dense = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )
        self.cat_dense = nn.Sequential(
            nn.Linear(num_categorical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )

        # Multi-head attention mechanism
        self.attention = MultiHeadAttentionBlock(num_ftrs + 32 + 64 + 64, num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(num_ftrs + 32 + 64 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def unfreeze_last_n_layers(self, model, n):
        """
        Unfreeze the last n layers of a model for fine-tuning.
        """
        params = list(model.parameters())
        for param in params[:-n]:
            param.requires_grad = False
        for param in params[-n:]:
            param.requires_grad = True

    def forward(self, image, mask, numerical, categorical):
        """
        Forward pass of the multimodal model.
        """
        # Process image features
        x_img = self.efficientnet(image)

        # Process mask features
        x_mask = self.mask_layers(mask[:, 0:1, :, :])
        x_mask = x_mask.view(mask.size(0), -1)

        # Process numerical and categorical feature
        x_num = self.num_dense(numerical)
        x_cat = self.cat_dense(categorical)

        # Combine all features
        combined = torch.cat((x_img, x_mask, x_num, x_cat), dim=1)

        # Apply multi-head attention
        combined = self.attention(combined)

        # Final fully connected layers
        x = F.relu(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
class ResMultimodalModel(nn.Module):
    """
    A multimodal neural network model that combines image, mask, numerical, and categorical data.
    This version uses ResNet-18 for image feature extraction.
    """
    def __init__(self, num_numerical_features, num_categorical_features, dropout_rate=0.5, num_heads=4):
        super(ResMultimodalModel, self).__init__()

        # Image feature extractor (ResNet-18)
        self.resnet = models.resnet18(pretrained=True)
        self.unfreeze_last_n_layers(self.resnet, 5)  # Unfreeze last 5 layers
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Mask feature extractor
        self.mask_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mask_bn = nn.BatchNorm2d(32)
        self.mask_layers = nn.Sequential(
            self.mask_conv, self.mask_bn, nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Numerical and categorical feature processors
        self.num_dense = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )
        self.cat_dense = nn.Sequential(
            nn.Linear(num_categorical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )

        # Multi-head attention mechanism
        self.attention = MultiHeadAttentionBlock(num_ftrs + 32 + 64 + 64, num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(num_ftrs + 32 + 64 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def unfreeze_last_n_layers(self, model, n):
        """
        Unfreeze the last n layers of a model for fine-tuning.
        """
        params = list(model.parameters())
        for param in params[:-n]:
            param.requires_grad = False
        for param in params[-n:]:
            param.requires_grad = True

    def forward(self, image, mask, numerical, categorical):
        """
        Forward pass of the multimodal model.
        """
        # Process image features
        x_img = self.resnet(image)

        # Process mask features
        x_mask = self.mask_layers(mask[:, 0:1, :, :])
        x_mask = x_mask.view(mask.size(0), -1)

        # Process numerical and categorical feature
        x_num = self.num_dense(numerical)
        x_cat = self.cat_dense(categorical)

        # Combine all features
        combined = torch.cat((x_img, x_mask, x_num, x_cat), dim=1)

        # Apply multi-head attention
        combined = self.attention(combined)

        # Final fully connected layers
        x = nn.ReLU()(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        x = nn.ReLU()(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class Res50MultimodalModel(nn.Module):
    """
    A multimodal neural network model that combines image, mask, numerical, and categorical data.
    This version uses ResNet-18 for image feature extraction.
    """
    def __init__(self, num_numerical_features, num_categorical_features, dropout_rate=0.5, num_heads=4):
        super(Res50MultimodalModel, self).__init__()

        # Image feature extractor (ResNet-50)
        self.resnet = models.resnet50(pretrained=True)
        self.unfreeze_last_n_layers(self.resnet, 5)  # Unfreeze last 5 layers
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Mask feature extractor
        self.mask_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mask_bn = nn.BatchNorm2d(32)
        self.mask_layers = nn.Sequential(
            self.mask_conv, self.mask_bn, nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Numerical and categorical feature processors
        self.num_dense = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )
        self.cat_dense = nn.Sequential(
            nn.Linear(num_categorical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )

        # Multi-head attention mechanism
        self.attention = MultiHeadAttentionBlock(num_ftrs + 32 + 64 + 64, num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(num_ftrs + 32 + 64 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def unfreeze_last_n_layers(self, model, n):
        """
        Unfreeze the last n layers of a model for fine-tuning.
        """
        params = list(model.parameters())
        for param in params[:-n]:
            param.requires_grad = False
        for param in params[-n:]:
            param.requires_grad = True

    def forward(self, image, mask, numerical, categorical):
        """
        Forward pass of the multimodal model.
        """
        # Process image features
        x_img = self.resnet(image)

        # Process mask features
        x_mask = self.mask_layers(mask[:, 0:1, :, :])
        x_mask = x_mask.view(mask.size(0), -1)

        # Process numerical and categorical feature
        x_num = self.num_dense(numerical)
        x_cat = self.cat_dense(categorical)

        # Combine all features
        combined = torch.cat((x_img, x_mask, x_num, x_cat), dim=1)

        # Apply multi-head attention
        combined = self.attention(combined)

        # Final fully connected layers
        x = nn.ReLU()(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        x = nn.ReLU()(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class InceptionMultimodalModel(nn.Module):
    """
    A multimodal neural network model that combines image, mask, numerical, and categorical data.
    This version uses InceptionV3 for image feature extraction.
    """

    def __init__(self, num_numerical_features, num_categorical_features, dropout_rate=0.5, num_heads=4):
        super(InceptionMultimodalModel, self).__init__()

        # Image feature extractor (InceptionV3)
        self.inception = models.inception_v3(pretrained=True)
        self.unfreeze_last_n_layers(self.inception, 5)  # Unfreeze last 5 layers
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Identity()

        # Mask feature extractor
        self.mask_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mask_bn = nn.BatchNorm2d(32)
        self.mask_layers = nn.Sequential(
            self.mask_conv, self.mask_bn, nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Numerical and categorical feature processors
        self.num_dense = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )
        self.cat_dense = nn.Sequential(
            nn.Linear(num_categorical_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(dropout_rate)
        )

        # Multi-head attention mechanism
        self.attention = MultiHeadAttentionBlock(num_ftrs + 32 + 64 + 64, num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(num_ftrs + 32 + 64 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def unfreeze_last_n_layers(self, model, n):
        """
        Unfreeze the last n layers of a model for fine-tuning.
        """
        params = list(model.parameters())
        for param in params[:-n]:
            param.requires_grad = False
        for param in params[-n:]:
            param.requires_grad = True

    def forward(self, image, mask, numerical, categorical):
        """
        Forward pass of the multimodal model.
        """
        # Process image features
        # InceptionV3 expects input size of 299x299
        if image.size(2) != 299 or image.size(3) != 299:
            image = F.interpolate(image, size=(299, 299), mode='bilinear', align_corners=False)

        x_img = self.inception(image)

        # If in training mode, InceptionV3 returns both output and aux_output
        if self.training and isinstance(x_img, tuple):
            x_img = x_img[0]  # Use only the main output, discard aux_output

        # Process mask features
        x_mask = self.mask_layers(mask[:, 0:1, :, :])
        x_mask = x_mask.view(mask.size(0), -1)

        # Process numerical and categorical feature
        x_num = self.num_dense(numerical)
        x_cat = self.cat_dense(categorical)

        # Combine all features
        combined = torch.cat((x_img, x_mask, x_num, x_cat), dim=1)

        # Apply multi-head attention
        combined = self.attention(combined)

        # Final fully connected layers
        x = F.relu(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TransferLearningModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Unfreeze more layers
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, masks):
        x = torch.cat([images, masks], dim=1)
        return self.resnet(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, mask=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        enc = self.encoder(x)
        mid = self.middle(enc)
        dec = self.decoder(mid)
        return dec


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
class VGGClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)