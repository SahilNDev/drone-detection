import pickle
import streamlit as st
from PIL import Image
import cv2

st.markdown(f"""<style>
         .stApp {{
             background-image: url();
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
)

st.title("Predict whether it is a bird or drone")

file = st.file_uploader("Upload the image here:", type=["jpg"])
if st.button("Submit"):
    with(file.name, "wb") as img_file:
        img_file.write(file.getbuffer())
    bytes_data = file.name
    st.image(bytes_data)
    item = cv2.resize(bytes_data, (100,100))
    nsamples,nx,ny = item.shape
    tester = item.reshape((nsamples,nx*ny))
    model = pickle.load(open("model.sav", "rb"))
    prediction = model.predict(tester)
    st.write(prediction)