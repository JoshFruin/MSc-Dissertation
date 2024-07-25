import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torchvision.models as models
from torch_geometric.utils import grid, add_self_loops
from transformers import ViTModel
from torch_geometric.data import Data as GeometricData


class ViTGNNHybrid(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTGNNHybrid, self).__init__()

        # Vision Transformer
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove the final classification head

        # GNN layers
        self.gcn1 = GCNConv(768, 512)
        self.gcn2 = GCNConv(512, 256)
        self.gcn3 = GCNConv(256, 128)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, images, edge_index):
        # Pass through ViT
        x = self.vit(images)

        # Create graph data structure
        batch_size = x.size(0)
        graph_data = GeometricData(x=x.view(-1, x.size(-1)), edge_index=edge_index)

        # Pass through GNN
        x = self.gcn1(graph_data.x, graph_data.edge_index)
        x = F.relu(x)
        x = self.gcn2(x, graph_data.edge_index)
        x = F.relu(x)
        x = self.gcn3(x, graph_data.edge_index)
        x = F.relu(x)

        # Reshape for classification
        x = x.view(batch_size, -1, x.size(-1)).mean(dim=1)

        # Classification
        x = self.classifier(x)
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