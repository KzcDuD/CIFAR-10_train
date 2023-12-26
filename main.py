import streamlit as st
import torch # load model and predict

st.title("Graphs recognition")
st.write("This is a simple web app to classify graphs")

# Add image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file :
    # Process the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

model = torch.load('./model/model.pt')