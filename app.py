import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import logging
from pathlib import Path
from typing import Tuple

# Config
CONFIG = {
    "model_paths": {
        "young2old": "models/G_AB_9.pth",
        "old2young": "models/G_BA_9.pth"
    },
    "input_shape": (3, 200, 200),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "residual_blocks": 3,
    "supported_formats": ["jpg", "jpeg", "png"]
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    """Residual block for the generator."""
    def __init__(self, in_features: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
    
class GeneratorResNet(nn.Module):
    """Generator model for face aging."""
    def __init__(self, input_shape: Tuple[int, int, int], num_residual_blocks: int = 9):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]
        out_features = 64
        
        # Initial layer
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        
        # Downsampling
        in_features = out_features
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            
        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
            
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            
        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
@st.cache_resource
def load_models():
    """Load and cache both transformation models."""
    try:
        models = {}
        for direction, path in CONFIG["model_paths"].items():
            model = GeneratorResNet(CONFIG["input_shape"], CONFIG["residual_blocks"]).to(CONFIG["device"])
            state_dict = torch.load(path, map_location=CONFIG["device"])
            model.load_state_dict(state_dict)
            model.eval()
            models[direction] = model
        return models
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        st.error("Error loading the models. Please check if model files exist.")
        raise

def process_image(image: Image.Image, model: GeneratorResNet) -> Image.Image:
    """Process an input image through the model."""
    transform = transforms.Compose([
        transforms.Resize(CONFIG["input_shape"][1:]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        with torch.no_grad():
            input_tensor = transform(image).unsqueeze(0).to(CONFIG["device"])
            output_tensor = model(input_tensor)
            
            output_image = output_tensor.squeeze(0).cpu().detach().numpy()
            output_image = (output_image + 1) / 2
            output_image = output_image.transpose(1, 2, 0)
            output_image = (output_image * 255).astype("uint8")
            return Image.fromarray(output_image)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        st.error("Failed to process the image. Please try another one.")
        return None

def main():
    st.set_page_config(page_title="Age Transformation App", page_icon="👤")
    
    st.title("Age Transformation App")
    st.write("Transform faces between young and old appearances.")

    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # Interface
    transform_direction = st.radio(
        "Select transformation direction:",
        ["Young to Old", "Old to Young"]
    )
    
    model_key = "young2old" if transform_direction == "Young to Old" else "old2young"
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=CONFIG["supported_formats"]
    )
    
    # Example images section
    st.write("Or select one of the example images below:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Old Example"):
            example_path = "examples/old.jpg"
            example_image = Image.open(example_path).convert("RGB")
            st.subheader("Original Image")
            st.image(example_image, use_container_width=True)
            st.subheader("Transformed Image")
            with st.spinner("Processing image..."):
                transformed_example = process_image(example_image, models[model_key])
                if transformed_example:
                    st.image(transformed_example, use_container_width=True)
    with col2:
        if st.button("Young Example"):
            example_path = "examples/young.jpg"
            example_image = Image.open(example_path).convert("RGB")
            st.subheader("Original Image")
            st.image(example_image, use_container_width=True)
            st.subheader("Transformed Image")
            with st.spinner("Processing image..."):
                transformed_example = process_image(example_image, models[model_key])
                if transformed_example:
                    st.image(transformed_example, use_container_width=True)

    if uploaded_file:
        try:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Original Image")
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_container_width=True)
            
            with col_b:
                st.subheader("Transformed Image")
                with st.spinner("Processing image..."):
                    transformed_image = process_image(image, models[model_key])
                    if transformed_image:
                        st.image(transformed_image, use_container_width=True)
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            st.error("Error processing the uploaded file. Please try again.")

if __name__ == "__main__":
    main()