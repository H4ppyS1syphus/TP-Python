import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from helper_tools.model import CNN
from helper_tools.dataset import OnTheFlyLicensePlateDataset
import urllib.request
import cv2
import numpy as np
import scipy.signal

# Set environment to avoid GUI issues
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #Crop 15% of the image from all sides
    image = image[int(image.shape[0] * 0.1):int(image.shape[0] * 0.9), int(image.shape[1] * 0.1):int(image.shape[1] * 0.9)]

    # Denoise the image pepper noise using median filter and gaussian filter and then apply bilateral filter
    denoised = cv2.medianBlur(image, 5)
    denoised = cv2.GaussianBlur(denoised, (5, 5), 0)
    denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    # Convert to grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # Invert the image (characters become white, background becomes black)
    inverted = cv2.bitwise_not(gray)

    # Apply lot of contrast then thresholding
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8, 8))
    contrast = clahe.apply(inverted)
    _, thresholded = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize to match CNN expected input shape (28, 196)
    processed = cv2.resize(thresholded, (196, 28), interpolation=cv2.INTER_NEAREST)

    # Normalize pixel values to range [0,1]
    processed = processed.astype(np.float32) / 255.0

    return processed


# Define function for making predictions
def predict_license_plate(image_tensor, model):
    """Runs the model on an image tensor and returns the predicted license plate characters."""
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 2)
        predicted = predicted.squeeze(0).cpu().numpy()

    return ''.join([label_to_char.get(p, '?') for p in predicted])

# Define function to save prediction results
def save_prediction_image(image, predicted_chars, filename):
    """Saves the predicted license plate image with its prediction text."""
    plt.figure(figsize=(10, 2))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f'Predicted License Plate: {predicted_chars}')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Load the dataset
test_loader = torch.utils.data.DataLoader(
    OnTheFlyLicensePlateDataset(num_samples=100, num_chars=7, font_path='fe_font/FE-FONT.TTF'),
    batch_size=1, shuffle=True
)

test_dataset = OnTheFlyLicensePlateDataset(num_samples=100, num_chars=7, font_path='fe_font/FE-FONT.TTF')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model_path = 'new_train.pth'
model = CNN(num_chars=7, num_classes=62)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)
model.eval()

# Character mapping for predictions
char_to_label = {str(i): i for i in range(10)}  # 0-9
char_to_label.update({chr(i + ord('A')): i + 10 for i in range(26)})  # A-Z
label_to_char = {v: k for k, v in char_to_label.items()}

# Number of characters expected in the plate
num_chars = 7

# Run predictions on 10 random samples from the dataset
with torch.no_grad():
    for i in range(10):
        image, label = test_dataset[i]
        image = image.unsqueeze(0)  # Add batch dimension
        predicted_chars = predict_license_plate(image, model)

        # Get true label
        true_chars = ''.join([label_to_char[l.item()] for l in label])

        # Save prediction
        save_prediction_image(image.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                              f'True: {true_chars} | Predicted: {predicted_chars}',
                              f'prediction_{i}.png')

# URL of the license plate image for online testing
url = 'https://evs-strapi-images-prod.imgix.net/changer_plaque_immatriculation_d235e7ed91.jpg?w=3840&q=75'

# Read the image from the URL
resp = urllib.request.urlopen(url)
image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

# Manually define cropping coordinates
x_start = 600
x_end = image.shape[1] - 600
y_start = 800
y_end = image.shape[0] - 800

# Crop and preprocess
cropped_image = image[y_start:y_end, x_start:x_end]
gray_resized = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
inverted_image = cv2.bitwise_not(gray_resized)
normalized_image = inverted_image.astype(np.float32) / 255.0
resized_image = cv2.resize(normalized_image, (28 * num_chars, 28), interpolation=cv2.INTER_AREA)

# Convert to tensor
tensor_image = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).to(device)

# Predict and save the result
predicted_chars = predict_license_plate(tensor_image, model)
save_prediction_image(normalized_image, predicted_chars, 'predicted_license_plate.png')

# **Evaluate additional plaques from dataset**
plaque_paths = [
    "data/plaques/plaque-001.png",
    "data/plaques/plaque-002.png",
    "data/plaques/plaque-003.png",
    "data/plaques/plaque-004.png",
    "data/plaques/plaque-005.png"
]

for plaque_path in plaque_paths:
    # Preprocess the plaque
    processed_image = preprocess_image(plaque_path)

    # Convert to tensor
    tensor_image = torch.from_numpy(processed_image).unsqueeze(0).unsqueeze(0).to(device)

    # Predict
    predicted_chars = predict_license_plate(tensor_image, model)

    # Save the evaluated image with the prediction
    save_prediction_image(processed_image, predicted_chars, f'predicted_{os.path.basename(plaque_path)}')

print("Evaluation completed. Predictions saved.")
