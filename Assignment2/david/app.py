
import streamlit as st
import tensorflow as tf
import streamlit as st
st.markdown('# Cheese/Chocolate Image Classifier')
model=tf.keras.models.load_model('best_model_upgrade.hdf5')


file = st.file_uploader("Please upload a picture containing either cheese or chocolate", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
size = (224,224)    

    
        

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    prediction_round = prediction.round()
    if prediction_round == 1:
        st.write("I think it contains chocolate")
    else:
        st.write("I think it contains cheese")
    st.write("Below you view the assigned probability to the Image containing chocolate")
    st.write(prediction)
