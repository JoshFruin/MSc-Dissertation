import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torchvision.models as models
from torch_geometric.utils import grid, add_self_loops
from transformers import ViTModel, ViTConfig
from torch_geometric.data import Data as GeometricData


class MultimodalBreastCancerModel(nn.Module):
    def __init__(self, num_tabular_features):
        super(MultimodalBreastCancerModel, self).__init__()

        # Image feature extractor (using a pre-trained ResNet)
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Identity()  # Remove the last fully connected layer

        # Tabular data processing
        self.tabular_model = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Combine image and tabular features
        self.combined_layer = nn.Sequential(
            nn.Linear(2048 + 32, 512),  # 2048 from ResNet, 32 from tabular data
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 2 output classes: benign and malignant
        )

    def forward(self, image, tabular_data):
        image_features = self.image_model(image)
        tabular_features = self.tabular_model(tabular_data)
        combined_features = torch.cat((image_features, tabular_features), dim=1)
        output = self.combined_layer(combined_features)
        return output

class ViTGNNHybrid(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ViTGNNHybrid, self).__init__()

        # Vision Transformer (ViT) part
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove the classification head

        # GNN part
        self.conv1 = GCNConv(768, 256)  # 768 is the output dimension of ViT
        self.conv2 = GCNConv(256, 128)

        # Final classification layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, images, masks):
        batch_size = images.size(0)

        # Process images through ViT
        vit_features = self.vit(images)  # Shape: (batch_size, 768)

        # Create graph data
        edge_index = self.get_edge_index(masks)

        # Create a batch of graphs
        batch = torch.arange(batch_size).repeat_interleave(vit_features.size(1))
        graph_data = Data(x=vit_features, edge_index=edge_index, batch=batch)

        # GNN layers
        x = self.conv1(graph_data.x, graph_data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, graph_data.edge_index)
        x = torch.relu(x)

        # Global pooling
        x = global_mean_pool(x, graph_data.batch)

        # Final classification
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_edge_index(self, masks):
        # This function should return the edge index based on the mask
        # For simplicity, we'll create a fully connected graph
        # You may want to implement a more sophisticated method based on your masks
        num_nodes = masks.size(1) * masks.size(2)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        return edge_index

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