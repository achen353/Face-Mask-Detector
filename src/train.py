from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Instantiate an argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d",
                    default='MFN', choices=['MFN', 'RMFD'],
                    help="dataset to train the model on")
args = parser.parse_args()

# Validate argument
if args.dataset != "MFN" and args.dataset != "RMFD":
    raise ValueError("Please provide a valid dataset choice: `MFN` or `RMFD`.")

# Change the working directory from src to root if needed
current_full_dir = os.getcwd()
print("Current working directory: " + current_full_dir)
if current_full_dir.split("/")[-1] == "src":
    root = current_full_dir[:-4]
    os.chdir(root)
    print("Changed working directory to: " + root)

# Initialize number of classes and labels
NUM_CLASS, class_names = None, None
if args.dataset == "MFN":
    NUM_CLASS = 3
    class_names = ['face_with_mask_incorrect', 'face_with_mask_correct', 'face_no_mask']
elif args.dataset == "RMFD":
    NUM_CLASS = 2
    class_names = ['face_with_mask', 'face_no_mask']

# Initialize the initial learning rate, number of epochs to train for, and batch size
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32
IMG_SIZE = 224
dataset_path = "./data/" + args.dataset + "/"
checkpoint_filepath = "./checkpoint_" + args.dataset + "/epoch-{epoch:02d}-val_acc-{val_accuracy:.4f}.h5"
model_save_path = "./mask_detector_models/mask_detector_" + args.dataset + ".h5"
figure_save_path = "./figures/train_plot_" + args.dataset + ".jpg"

print("Num of classes: " + str(NUM_CLASS))
print("Classes: " + str(class_names))
print("Dataset path: " + dataset_path)
print("Checkpoint: " + checkpoint_filepath)
print("Figure save path: " + figure_save_path)

# Construct the training/validation image generator for data augmentation
data_generator = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=preprocess_input,
    validation_split=0.2)

# Set as training data
train_generator = data_generator.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    subset='training')

# Set as validation data
validation_generator = data_generator.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    subset='validation')

# Load the pre-trained model and remove the head FC layer
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

# Construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(NUM_CLASS, activation="softmax")(head_model)

# Place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=base_model.input, outputs=head_model)

# Loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in base_model.layers:
    layer.trainable = False

# Compile our model
print("[INFO] compiling model...")
opt = Adam(lr=LEARNING_RATE)
if args.dataset == "MFN":
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
elif args.dataset == "RMFD":
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Add early stopping criterion
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.0001,
    patience=3,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True)

# Add model checkpoint
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=False,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='auto')

# Train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    callbacks=[early_stopping, checkpoint],
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=EPOCHS)

# Save best model
model.save(model_save_path)

# Create classification report
prediction = model.predict_generator(
    generator=validation_generator,
    verbose=1)
y_pred = np.argmax(prediction, axis=1)
print("Classification Report:")
print(classification_report(validation_generator.classes, y_pred, target_names=class_names))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(H.history["val_loss"])), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(H.history["accuracy"])), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(H.history["val_accuracy"])), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(figure_save_path)
