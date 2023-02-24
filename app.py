import cv2
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="IPO", layout="centered", initial_sidebar_state = "auto")

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright

def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img

def contrast_image(image, amount):
    img_contrast = cv2.convertScaleAbs(image, alpha=amount)
    return img_contrast


def main_loop():
    st.header("Image Processing using python OpenCV")
    st.sidebar.header("APPLY IMAGE FILTERS:")

    grayscale_filter = st.sidebar.checkbox('Convert to Grayscale')
    hsv_filter = st.sidebar.checkbox('Convert to HSV')
    enhance_details_filter = st.sidebar.checkbox('Enhance Details (Filters)')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)
    processed_image = original_image

    if grayscale_filter:
        processed_image = grayscale(original_image)
    if hsv_filter:
        processed_image = hsv(original_image)

    if enhance_details_filter:
        filters = st.sidebar.multiselect("Choose Filters", ("Blur", "Brightness", "Contrast"))
        if "Blur" in filters:
            blur_amount = st.sidebar.slider("Blur Amount", min_value=0.0, max_value=3.5)
            processed_image = blur_image(original_image, blur_amount)
        if "Brightness" in filters:
            brightness_amount = st.sidebar.slider("Brightness Amount", min_value=-50, max_value=50, value=0)
            processed_image = brighten_image(original_image, brightness_amount)
        if "Contrast" in filters:
            contrast_amount = st.sidebar.slider("Contrast Amount", min_value=-3.5, max_value=3.5)
            processed_image = contrast_image(original_image, contrast_amount)

    org_img = plt.figure(figsize=(15, 10))
    plt.hist(original_image.ravel(), 256, [0,256], color='green')
    plt.ylabel("Number Of Pixels", color='green')
    plt.xlabel("Pixel Intensity- From 0-255", color='green')
    plt.title("Histogram Showing Pixel Intensity And Corresponding Number Of Pixels", color='green')

    st.subheader("Original Image")
    st.image(original_image, use_column_width=True)
    st.pyplot(org_img)

    proc_img = plt.figure(figsize = (15, 10))
    plt.hist(processed_image.ravel(), 256, [0,256], color='blue')
    plt.ylabel("Number Of Pixels", color='blue')
    plt.xlabel("Pixel Intensity- From 0-255", color='blue')
    plt.title("Histogram Showing Pixel Intensity And Corresponding Number Of Pixels", color='blue')

    st.subheader("Processed Image")
    st.image(processed_image, use_column_width=True)
    st.pyplot(proc_img)

if __name__ == '__main__':
    main_loop()
