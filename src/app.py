from tensorflow.keras.models import load_model
from detect_mask_image import detect_mask
import streamlit as st
import cv2
import numpy as np
import os


def main():
    """Face Detection App"""
    st.title("Face Mask Detector")
    st.text("Build with Streamlit, OpenCV and Tensorflow")

    # Change the working directory from src to root if needed
    current_full_dir = os.getcwd()
    print("Current working directory: " + current_full_dir)
    if current_full_dir.split("/")[-1] == "src":
        root = current_full_dir[:-4]
        os.chdir(root)
        print("Changed working directory to: " + root)

    # Initialize model save path
    mask_detector_model_path = "./mask_detector_models/mask_detector_MFN.h5"
    face_confidence = 0.5

    # Load the face detector model from disk
    print("[INFO] loading face detector model...")
    prototxt_path = "./face_detector_model/deploy.prototxt"
    weights_path = "./face_detector_model/res10_300x300_ssd_iter_140000.caffemodel"
    face_detector = cv2.dnn.readNet(prototxt_path, weights_path)

    # Load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    mask_detector = load_model(mask_detector_model_path)

    activities = ["Detect on Image", "Detect through Webcam", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Detect on Image":
        st.subheader("Mask Detection on Image")
        img_file = st.file_uploader("Upload a image", type=['jpg', 'png', 'jpeg'])

        if img_file is not None:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.text("Original Image")
            st.image(img, caption="Image successfully uploaded.", channels="BGR")
            if st.button('Process'):
                out_img = detect_mask(img, face_detector, mask_detector, face_confidence, False)
                st.image(out_img, use_column_width=True)

        if choice == "Detect through Webcam":
            st.subheader("Mask Detection on Webcam")
            st.text("This feature will be available soon.")


if __name__ == '__main__':
    main()
