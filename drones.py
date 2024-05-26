import pickle
import streamlit as st
from PIL import Image
import cv2
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://github.com/Divyam-kr/Files-for-streamlit-design-elements/blob/main/ezgif.com-video-to-gif.gif?raw=true");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

custom_style = """
<style>
   .custom-text {
    background-color: black;  /* Set background color to black */
    color: white;  /* Set text color to white */
    padding: 10px;  /* Add some padding for readability */
    font-size: 50px;  /* Increase font size */
    font-weight: bold;  /* Make the text bold */
    font-family: 'Arial', sans-serif;  /* Set font family */
    border-radius: 5px;  /* Optional: rounded corners */
}
</style>
"""
# Display the custom style
st.markdown(custom_style, unsafe_allow_html=True)
st.title("Predict whether it is a bird or drone: ")
tab1 , tab2 = st.tabs(['Model', 'About Us'])
with tab1:
         st.header("Model Alpha")
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
             st.write(f'<div class="custom-text">{dict1[prediction[0]]}</div>', unsafe_allow_html=True)
         
with tab2:
    st.header("About Us")
    st.markdown("The Creators of the Website:")
    st.write("1. Bhavya Dashottar")
    st.write("2. Divyam Kumar")
    
         
