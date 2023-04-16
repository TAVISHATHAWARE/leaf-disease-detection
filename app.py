import streamlit as st
from PIL import Image, ImageOps
import numpy as np

st.title("Leaf Disease Detection Model")
st.header("")
st.text("Upload a leaf image for image classification as healthy or unhealthy")

from image_classification import img_classification

uploaded_file = st.file_uploader("Upload a leaf image for disease detection...", type=["jpg","png"])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = img_classification(image, 'model.h5')
        class_names=['Unhealthy Tomato Bacterial spot','Unhealthy Tomato Early blight','Unhealthy Tomato Late blight',
                     'Unhealthy Tomato Leaf Mold','Unhealthy Tomato Septoria leaf spot',
                     'Unhealthy Tomato Spider mites Two-spotted spider mite','Unhealthy Tomato Target Spot',
                     'Unhealthy Tomato Yellow Leaf Curl Virus','Unhealthy Tomato mosaic virus','Healthy Tomato', 
                     'Unhealthy Cauliflower alternia leaf spot','Unhealthy Cauliflower aphid colony','Unhealthy Cauliflower black leg',
                     'Unhealthy Cauliflower bugs attack','Unhealthy Cauliflower downy mildew','Healthy Cauliflower','Unhealthy Mango',
                     'Healthy Mango','Unhealthy Cauliflower ring spot','Unhealthy Cauliflower white rust']
        string = "This leaf is predicted as:" + class_names[label]
        st.success(string)
        if label<0.6:
          category="Healthy"
        elif label>0.6:
          category="Unhealthy"
    
        st.write(category)