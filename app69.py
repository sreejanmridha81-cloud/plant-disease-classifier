import pickle
import streamlit as st
import numpy as np
from PIL import Image
import os

if __name__ == "__main__":
    # Pull the port assigned by Render, defaulting to 10000 locally
    port = int(os.environ.get("PORT", 10000))
    # Must use 0.0.0.0 so the proxy gateway can route traffic to it
    app.run(host="0.0.0.0", port=port)
class_indices=pickle.load(open('class_indices.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
def load_and_preprocess_image(image_path,target_size=(224,224)):
  img=Image.open(image_path)
  img=img.resize(target_size)
  img_array=np.array(img)
  #add batch dimension
  img_array=np.expand_dims(img_array,axis=0)
  img_array=img_array.astype('float32')/256
  return img_array
#function to classify the image
def predict_image_class(model,image_path,class_indices):  
  preprocessed_img=load_and_preprocess_image(image_path)
  prediction=model.predict(preprocessed_img)

  prediction_class_index = np.argmax(prediction, axis=1)[0]
  
  predicted_plant_name = class_indices[int(prediction_class_index)]
  return predicted_plant_name


st.title("🌿Plant Disease Classifier🌿")
upload_image=st.file_uploader('upload an image....',type=['jpg','jpeg','png'])
if upload_image is not None: 
    # 1. Display the image
    st.image(upload_image, caption='Uploaded Image') 
    
    # 2. Run prediction
    result = predict_image_class(model, upload_image, class_indices)
    
    # 3. Print result
    st.success(f"Predicted Plant-Disease: {result}")
