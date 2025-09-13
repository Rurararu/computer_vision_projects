import streamlit as st
from PIL import Image

from util import classify, set_background

set_background('./bgs/bg1.jpg')

# set title
st.title('Pneumonia classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])

model_path = "pneumonia_classifier_model.pth"
class_names = ["NORMAL", "PNEUMONIA"]

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width=True)
    class_name, conf_score = classify(image, model_path, class_names)
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))