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
    st.image(file)
    st.write(file)
    item = cv2.resize(file, (100,100))
    nsamples,nx,ny = item.shape
    tester = item.reshape((nsamples,nx*ny))
    model = pickle.load(open("model.sav", "rb"))
    prediction = model.predict(tester)
    st.write(prediction)
