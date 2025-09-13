import base64

import streamlit as st
import torch.nn as nn
from torchvision import transforms, models
import torch



def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model_path, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model_path (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)

    # make prediction
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2)  # pneumonia / normal

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()


    # index = np.argmax(prediction)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence_score = probabilities[0][predicted_class].item()

    return class_names[predicted_class], confidence_score