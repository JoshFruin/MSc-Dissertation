import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom

class GradCAM:
    """
    Implements the Grad-CAM (Gradient-weighted Class Activation Mapping) technique.

    This class allows for the visualization of important regions in the input image
    that contribute to the model's decision for a specific class.

    Attributes:
        model (torch.nn.Module): The neural network model to analyze.
        target_layer (torch.nn.Module): The target layer in the model for Grad-CAM visualization.
        gradients (torch.Tensor): Stores the gradients of the target layer.
        activation (torch.Tensor): Stores the forward pass activation of the target layer.
    """
    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM object.

        Args:
            model (torch.nn.Module): The model to analyze.
            target_layer (torch.nn.Module): The target layer for Grad-CAM.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activation = None
        self.register_hooks()

    def register_hooks(self):
        """
            Registers the forward and backward hooks on the target layer.

            These hooks are used to capture the layer's activations (forward hook)
            and gradients (backward hook) during the forward and backward passes.
            """
        def forward_hook(module, input, output):
            self.activation = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_data, target_class=None):
        """
        Generates the Grad-CAM heatmap for the given input data.

        Args:
            input_data (tuple): A tuple containing (image, mask, numerical, categorical) data.
            target_class (int, optional): The target class for visualization. If None, uses the predicted class.

        Returns:
            numpy.ndarray: The generated Grad-CAM heatmap.
        """
        image, mask, numerical, categorical = input_data
        output = self.model(image, mask, numerical, categorical)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute the channel-wise mean of the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activation.shape[1]):
            self.activation[:, i, :, :] *= pooled_gradients[i]

        # Compute the mean over the channels to get the heatmap
        heatmap = torch.mean(self.activation, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU operation
        heatmap /= np.max(heatmap)  # Normalize to [0, 1]
        return heatmap

def apply_colormap(heatmap, original_img):
    """
    Applies a color map to the heatmap and overlays it on the original image.

    Args:
        heatmap (numpy.ndarray): The Grad-CAM heatmap.
        original_img (PIL.Image): The original input image.

    Returns:
        PIL.Image: The image with the colored heatmap overlaid.
    """
    # Resize heatmap to match original image size
    heatmap_resized = zoom(heatmap, (original_img.size[1] / heatmap.shape[0], original_img.size[0] / heatmap.shape[1]))

    # Convert heatmap to RGB colormap
    heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
    colored_heatmap = plt.cm.jet(heatmap_resized)[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

    # Overlay the heatmap on the original image
    original_img_array = np.array(original_img.convert('RGB'))
    overlayed_img = original_img_array * 0.7 + colored_heatmap * 0.3
    overlayed_img = overlayed_img.astype(np.uint8)

    return Image.fromarray(overlayed_img)

def visualize_gradcam(model, image, mask, numerical, categorical, target_layer):
    """
    Visualizes the Grad-CAM result for a given input.

    This function generates and displays the Grad-CAM heatmap alongside
    the original image for comparison.

    Args:
        model (torch.nn.Module): The model to analyze.
        image (torch.Tensor): The input image tensor.
        mask (torch.Tensor): The input mask tensor.
        numerical (torch.Tensor): The input numerical features tensor.
        categorical (torch.Tensor): The input categorical features tensor.
        target_layer (torch.nn.Module): The target layer for Grad-CAM visualization.
    """
    model.eval()
    gradcam = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap = gradcam.generate_cam((image, mask, numerical, categorical))

    # Convert tensor to PIL Image for visualization
    original_img = Image.fromarray((image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    # Apply colormap and overlay
    cam_image = apply_colormap(heatmap, original_img)

    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(original_img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(cam_image)
    ax2.set_title('Grad-CAM')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()