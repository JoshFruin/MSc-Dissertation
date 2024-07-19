import torch
import torch.nn as nn
import torchvision.models as models
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
import traceback

import torch
import torch.nn as nn
import torchvision.models as models
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class ViTGNNHybrid(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTGNNHybrid, self).__init__()

        # ViT Feature Extractor
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove the classification head

        # Adapt ViT for grayscale input
        self.vit.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))

        # Graph Construction Layer
        self.graph_constructor = nn.Linear(768, 128)

        # GNN Layers
        self.conv1 = GCNConv(128, 64)
        self.conv2 = GCNConv(64, 32)

        # Final classification layer
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x, mask):
        batch_size = x.size(0)

        # ViT Feature Extraction
        x = self.vit(x)  # Shape: (batch_size, 768)

        # Convert mask to float and use it to focus on ROI
        mask_float = mask.float()
        mask_pooled = nn.functional.adaptive_avg_pool2d(mask_float, (1, 1)).squeeze()
        x = x * mask_pooled.unsqueeze(1)

        # Graph Construction
        node_features = self.graph_constructor(x)  # Shape: (batch_size, 128)

        # Create a batch of fully connected graphs
        edge_index = torch.stack([torch.arange(batch_size).repeat_interleave(batch_size),
                                  torch.arange(batch_size).repeat(batch_size)])

        # Create a PyG Data object
        data = Data(x=node_features, edge_index=edge_index)

        # GNN Layers
        x = torch.relu(self.conv1(data.x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))

        # Global Pooling
        x = global_mean_pool(x, torch.arange(batch_size))

        # Classification
        x = self.classifier(x)

        return x

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