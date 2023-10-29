import streamlit as st
import numpy as np
import cv2
from inference import Inference
import cv2
from imread_from_url import imread_from_url
If = Inference()

st.title("Detect Image")

# Option to upload an image file
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Option to enter an image URL
image_url = st.text_input("Or enter an image URL")


def resize_image(image, target_width, target_height):
    # Get the original dimensions
    height, width, _ = image.shape

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Calculate the new dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    # Resize the image using OpenCV
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    return resized_image


# Process the image based on the chosen method
if uploaded_image:
    image_np = cv2.imdecode(
        np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR
    )
    st.image(image_np, channels="BGR", caption="Uploaded Image", use_column_width=True)
    resized_image = resize_image(image_np, 640, 640)
    st.image(If.Detect(resized_image), caption="Detected Image", use_column_width=True)

elif image_url:
    img_url = image_url
    img = imread_from_url(img_url)
    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)
    resized_image = resize_image(img, 640, 640)
    st.image(If.Detect(resized_image), caption="Detected Image", use_column_width=True)
