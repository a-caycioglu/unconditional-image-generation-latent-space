import argparse
from pipeline import MyPipeline
import torch

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate images using MyPipeline.")
parser.add_argument('--model_id', type=str, default="output_dreambooth", help='The ID of the base model to load. It will be the path of "outputs" folder.')
parser.add_argument('--num_images', type=int, default=10, help='Number of images to generate.')
args = parser.parse_args()

# Assign the model_id from the parsed arguments
model_id = args.model_id

torch.manual_seed(111) # It is used to compare results
# Determine the device to use
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load the model
pipeline = MyPipeline.from_pretrained(model_id).to(device)

# Generate and save images
for i in range(args.num_images):
    image = pipeline(num_inference_steps=100).images[0]
    # Save the image
    image.save("generated_dreambooth_" + str(i) + ".png")
