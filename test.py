import timm
import torch

# Create model
model = timm.create_model('swinv2_base_window16_256.ms_in1k', pretrained=True)

# Access the layers
layers = model.layers

# Create dummy input (batch_size=1, channels=3, height=256, width=256)
x = torch.randn(1, 3, 256, 256)

# First apply patch embedding
x = model.patch_embed(x)
print("After patch_embed:", x.shape)

# Run through each layer and print shape
for i, layer in enumerate(layers):
    x = layer(x)
    print(f"After layer {i+1}:", x.shape)