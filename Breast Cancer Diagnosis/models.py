import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import vit_b_16
import torch_geometric.nn as pyg_nn
from transformers import ViTModel
import dgl
import dgl.nn.pytorch as dglnn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):  # Added in_channels parameter
        super(ViTClassifier, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        # Modify first layer for single-channel input
        self.vit.conv_proj = nn.Conv2d(in_channels, self.vit.hidden_dim, kernel_size=(16, 16), stride=(16, 16))
        # Update classifier head
        in_features = self.vit.heads[0].in_features
        self.vit.heads = nn.Sequential(
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.vit(x)


class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print(f"Input x shape: {x.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Batch shape: {batch.shape}")

        x = x.view(-1, 3)
        print(f"Reshaped x shape: {x.shape}")

        x = F.relu(self.conv1(x, edge_index))
        print(f"After conv1 shape: {x.shape}")
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        print(f"After conv2 shape: {x.shape}")
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        print(f"After conv3 shape: {x.shape}")
        x = global_mean_pool(x, batch)
        print(f"After global_mean_pool shape: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        print(f"Final output shape: {x.shape}")

        return x


class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)  # Replace with your ViT model
        self.fc = nn.Linear(1000, num_classes)  # Assuming ViT outputs 1000 features

    def forward(self, x, edge_index=None, batch=None):
        if x.dim() == 2:  # If input is 2D, reshape it to 4D
            n = int(torch.sqrt(torch.tensor(x.shape[0])))
            x = x.view(1, 1, n, n)  # Assuming input is square and grayscale
        elif x.dim() == 3:  # If input is 3D, reshape it to 4D
            x = x.unsqueeze(0)
        vit_out = self.vit(x)
        out = self.fc(vit_out)
        return out

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