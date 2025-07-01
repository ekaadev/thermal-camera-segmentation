import streamlit as st
import numpy as np
import cv2 as cv

def canny_edge_detection(image):
    """
    Perform Canny edge detection on the input image.
    """
    edges = cv.Canny(image, 100, 200)
    return edges

def segmentation_with_treshold(image):
    """
    Perform segmentation using thresholding on the input image.
    """
    _, thresh = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    
    return mask

st.title("Thermal Camera Segmentation App")

# with open camera
enable = st.checkbox("Enable Camera", value=False)

if not enable:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture", key="camera_input", disabled=not enable)

if uploaded_file is not None:
    # convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

    img = cv.cvtColor(opencv_image, cv.COLOR_BGR2GRAY)
    assert img is not None, "file could not be read, check with os.path.exists()"

    edges = segmentation_with_treshold(img)

    # display the all images
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Original Image", use_container_width=True)

    with col2:
        st.image(edges,  caption="Segmentation Image", use_container_width=True)

