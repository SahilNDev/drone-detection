import pickle
import streamlit as st
from PIL import Image
import cv2
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://github.com/Divyam-kr/Files-for-streamlit-design-elements/blob/main/Drone.gif?raw=true");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

st.title("Predict whether it is a bird or drone: ")

file = st.file_uploader("Upload the image here:", type=["jpg", "jpeg","png"])
if st.button("Submit"):
    with open(file.name, "wb") as image:
         image.write(file.getbuffer())
    bytes_data = file.name
    st.image(bytes_data)
    img_new = cv2.imread(bytes_data,0)
    item = cv2.resize(img_new, (100,100))
    nx,ny = item.shape
    tester = item.reshape((1,nx*ny))
    model = pickle.load(open("model.sav", "rb"))
    prediction = model.predict(tester)
    dict1 = {0:'Bird',1:'Drone'}
    st.write(dict1[prediction[0]])
