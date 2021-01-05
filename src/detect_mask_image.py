from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os


def detect_mask(img, face_detector, mask_detector, confidence_threshold, image_show=True):
    # Initialize the labels and colors for bounding boxes
    num_class = mask_detector.layers[-1].get_output_at(0).get_shape()[-1]
    labels, colors = None, None
    if num_class == 3:
        labels = ["Face with Mask Incorrect", "Face with Mask Correct", "Face without Mask"]
        colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]
    elif num_class == 2:
        labels = ["Face with Mask", "Face without Mask"]
        colors = [(0, 255, 0), (0, 0, 255)]

    # Load the input image from disk, clone it, and grab the image spatial dimensions
    (h, w) = img.shape[:2]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Record status
    # MFN: 0 is "mask correctly" and 1 is "no mask"
    # RMFD: 0 is "mask correctly", 1 is "mask incorrectly", and 2 is "no mask"
    status = 0

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > confidence_threshold:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = img[start_y:end_y, start_x:end_x]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # Pass the face through the model to determine if the face has a mask or not
            prediction = mask_detector.predict(face)[0]
            label_idx = np.argmax(prediction)

            # Determine the class label and color we'll use to draw the bounding box and text
            label = labels[label_idx]
            color = colors[label_idx]

            # Update the status
            if num_class == 3:
                if label_idx == 0:
                    temp = 1
                elif label_idx == 1:
                    temp = 0
                else:
                    temp = 2
                status = max(status, temp)
            elif num_class == 2:
                status = max(status, label_idx)

            # Include the probability in the label
            label = "{}: {:.2f}%".format(label, max(prediction) * 100)

            # Display the label and bounding box rectangle on the output frame
            cv2.putText(img, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, 2)
        else:
            break

    if image_show is True:
        # Show the output image
        cv2.imshow("Output", img)
        cv2.waitKey(0)

    return status, img


def main():
    # Instantiate an argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True,
                        help="path to input image")
    parser.add_argument("--model", "-m", type=str,
                        default='MFN', choices=['MFN', 'RMFD'],
                        help="face mask detector model")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="minimum probability to filter weak face detections")
    args = parser.parse_args()

    # Change the working directory from src to root if needed
    current_full_dir = os.getcwd()
    print("Current working directory: " + current_full_dir)
    if current_full_dir.split("/")[-1] == "src":
        root = current_full_dir[:-4]
        os.chdir(root)
        print("Changed working directory to: " + root)

    # Validate arguments
    if not os.path.isfile(args.image):
        raise ValueError("Please provide a valid path relative to the root directory.")
    if args.model != "MFN" and args.model != "RMFD":
        raise ValueError("Please provide a valid model choice: `MFN` or `RMFD`.")
    if args.confidence > 1 or args.confidence < 0:
        raise ValueError("Please provide a valid confidence value between 0 and 1 (inclusive).")

    # Initialize model save path
    mask_detector_model_path = "./mask_detector_models/mask_detector_" + args.model + ".h5"
    confidence_threshold = args.confidence
    print("Mask detector save path: " + mask_detector_model_path)
    print("Face detector thresholding confidence: " + str(confidence_threshold))

    # Load the face detector model from disk
    print("[INFO] loading face detector model...")
    prototxt_path = "./face_detector_model/deploy.prototxt"
    weights_path = "./face_detector_model/res10_300x300_ssd_iter_140000.caffemodel"
    face_detector = cv2.dnn.readNet(prototxt_path, weights_path)

    # Load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    mask_detector = load_model(mask_detector_model_path)

    # Read the image and detect mask
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError("Your file type is not supported.")

    detect_mask(img, face_detector, mask_detector, confidence_threshold)


if __name__ == '__main__':
    main()
