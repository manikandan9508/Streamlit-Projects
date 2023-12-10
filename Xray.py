import streamlit as st
import os
import  cv2
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from random import shuffle , seed
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications   import EfficientNetB3
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input ,concatenate, Dense,Flatten , Activation,Conv2D ,Dropout ,MaxPooling2D ,GlobalAveragePooling2D,BatchNormalization
from skmultilearn.model_selection import iterative_train_test_split

st.write(""" 
# Deep Learning for detecting lung diseases from X-ray images  """)

# Load the CSV file
df = pd.read_csv("sample_labels.csv")

# Add a checkbox to show/hide the DataFrame
show_df = st.checkbox("Show DataFrame")

# Display the DataFrame if the checkbox is checked
if show_df:
    st.dataframe(df.head())

# Add a checkbox to show/hide the value counts
show_value_counts = st.checkbox("Show Value Counts")

# Display the value counts if the checkbox is checked
if show_value_counts:
    st.write("Value Counts for 'Finding Labels':")
    st.write(df["Finding Labels"].value_counts())

# Add a checkbox to show/hide the count plot for Patient Gender
show_gender_plot = st.checkbox("Show Patient Gender Count Plot")
if show_gender_plot:
    st.write("Patient Gender Count Plot:")
    sns.countplot(df["Patient Gender"])
    st.pyplot()

# Add a checkbox to show/hide the count plot for Finding Labels (top 20)
show_labels_plot = st.checkbox("Show Finding Labels Count Plot (Top 20)")
if show_labels_plot:
    st.write("Finding Labels Count Plot (Top 20):")
    plt.figure(figsize=(20, 20))
    sns.countplot(df["Finding Labels"][:20])
    st.pyplot()

# Prepare labels
Labels_before_pre = df["Finding Labels"]
Labels_after_pre = []

for i in range(len(Labels_before_pre)):
    split_labels = Labels_before_pre[i].split("|")
    if len(split_labels) == 1:
        Labels_after_pre.append(split_labels)
    else:
        lab = []
        for j in range(len(split_labels)):
            lab.append(split_labels[j])
        Labels_after_pre.append(lab)

# Add a checkbox to show/hide the prepared labels
show_prepared_labels = st.checkbox("Show Prepared Labels")

# Display the prepared labels if the checkbox is checked
if show_prepared_labels:
    st.write("Prepared Labels:")
    st.write(Labels_after_pre)

# Read X-rays
image_size = 256
image_file_path = "C:/Users/manik/PycharmProjects/pythonfile/streamlit/images/"
Labels_before_pre = df["Finding Labels"]
Labels_after_pre = []

scans = []

for i in tqdm(range(len(df["Image Index"]))):
    image = cv2.imread(image_file_path + df["Image Index"][i])
    if len(image.shape) > 2:
        resize_image = cv2.resize(image, (image_size, image_size))
        scans.append(resize_image[:, :, :3])  # Keep only the first three channels
    else:
        image = np.dstack([image] * 3)
        resize_image = cv2.resize(image, (image_size, image_size))
        scans.append(resize_image)

    # Prepare labels
    split_labels = Labels_before_pre[i].split("|")
    if len(split_labels) == 1:
        Labels_after_pre.append(split_labels)
    else:
        lab = []
        for j in range(len(split_labels)):
            lab.append(split_labels[j])
        Labels_after_pre.append(lab)

# Add a checkbox to show/hide the processed X-ray images and labels
show_images_and_labels = st.checkbox("Show Images and Labels")

# Display the processed X-ray images and labels if the checkbox is checked
if show_images_and_labels:
    st.write("Processed X-ray Images and Labels:")
    for i in range(len(scans)):
        st.image(scans[i], caption=f"Labels: {Labels_after_pre[i]}", use_column_width=True)

# Function to show images
def image_show(data, labels, number_of_image):
    # To generate random numbers
    numbers = np.random.randint(0, len(data), number_of_image)
    plt.figure(figsize=(40, 20))
    j = number_of_image / 10
    for _, i in enumerate(numbers):
        plt.subplot(j, 10, _ + 1)
        plt.imshow(data[i], cmap="gray")
        label = ""
        for x in labels[i]:
            label += x + " , "

        plt.title(label + "\n" + f"size {data[i].shape}")
        # To remove the number that appears around the image
        plt.xticks([]), plt.yticks([])
    st.pyplot()

# Add a checkbox to show/hide the images
show_images = st.checkbox("Show Images")

# Display the images if the checkbox is checked
if show_images:
    st.write("Generated Images:")
    image_show(scans, Labels_after_pre, number_of_image=10)


# Dictionary for annotation labels
classes = {
    0: "Hernia",
    1: "Pneumonia",
    2: "Fibrosis",
    3: "Edema",
    4: "Emphysema",
    5: "Cardiomegaly",
    6: "Pleural_Thickening",
    7: "Consolidation",
    8: "Pneumothorax",
    9: "Mass",
    10: "Nodule",
    11: "Atelectasis",
    12: "Effusion",
    13: "Infiltration",
    14: "No Finding",
}

# Function to get class label
def get_class(code):
    return classes[code]

# Function to get code for a given label
def get_code(labels):
    for key, value in classes.items():
        if value == labels:
            return key

# Add a checkbox to show/hide the dictionary and functions
show_info = st.checkbox("Show Dictionary and Functions")

# Display the dictionary and functions if the checkbox is checked
if show_info:
    st.write("Annotation Labels Dictionary:")
    st.write(classes)

    st.write("\nExample Usage:")
    st.write("Class code for 'Hernia':", get_code("Hernia"))
    st.write("Class label for code 0:", get_class(0))

# Add a checkbox to show/hide the converted labels and arrays
show_conversion_info = st.checkbox("Show Label Conversion Information")

# Display the information if the checkbox is checked
if show_conversion_info:
    st.write("Converted Labels:")
    st.write(Labels_after_pre)

    # Convert labels to LabelEncoder
    for i in range(len(Labels_after_pre)):
        Labels_after_pre[i] = [get_code(x) for x in Labels_after_pre[i]]

    st.write("Labels after Label Encoder:")
    st.write(Labels_after_pre)

    # Convert labels to one-hot-encoder with MultiLabelBinarizer from sklearn
    mlp=MultiLabelBinarizer()
    Labels=mlp.fit_transform(Labels_after_pre)

    st.write("Labels after One-Hot Encoder:")
    st.write(Labels)

    # Convert scans and labels to array
    scans=np.array(scans)
    Labels=np.array(Labels)

    st.write("Scans Shape:", scans.shape)
    st.write("Labels Shape:", Labels.shape)

# Split the data
X_train, y_train, X_test, y_test = iterative_train_test_split(scans,Labels, test_size=0.2)
X_val, y_val, X_test, y_test = iterative_train_test_split(X_test, y_test, test_size=0.7)

# Add a checkbox to show/hide the data shapes
show_data_shapes = st.checkbox("Show Data Shapes")

# Display the data shapes if the checkbox is checked
if show_data_shapes:
    st.write("Data Shapes:")
    st.write("X_train shape:", X_train.shape)
    st.write("y_train shape:", y_train.shape)
    st.write("X_val shape:", X_val.shape)
    st.write("y_val shape:", y_val.shape)
    st.write("X_test shape:", X_test.shape)
    st.write("y_test shape:", y_test.shape)

# Data generator
generator = ImageDataGenerator(
    rescale=1/255.0,
    samplewise_std_normalization=True,
    samplewise_center=True,
    rotation_range=90
)

batch_size = 16
train_generator = generator.flow(X_train, y_train, batch_size=batch_size)
val_generator = generator.flow(X_val, y_val, batch_size=batch_size)
test_generator = generator.flow(X_test, y_test, batch_size=batch_size)

# Add a checkbox to show/hide the generated images
show_generated_images = st.checkbox("Show Generated Images")

# Display the generated images if the checkbox is checked
if show_generated_images:
    st.write("Generated Images:")
    train_scans, train_labels = train_generator.__getitem__(0)
    for i in range(len(train_scans)):
        st.image(train_scans[i], caption=f"Labels: {train_labels[i]}", use_column_width=True)



image_size = 256
Inputs = Input((image_size, image_size, 3))

#1
c1=Conv2D(64 , (3,3) , activation="relu" , padding="same")(Inputs)
c1=Conv2D(64 , (3,3) , activation="relu" , padding="same")(c1)
c1=Conv2D(64 , (3,3) , activation="relu" , padding="same")(c1)
p1=MaxPooling2D(pool_size=(3,3))(c1)
#2
c2=Conv2D(128 , (3,3) , activation="relu" , padding="same")(p1)
c2=Conv2D(128 , (3,3) , activation="relu" , padding="same")(c2)
c2=Conv2D(128 , (3,3) , activation="relu" , padding="same")(c2)
p2=MaxPooling2D(pool_size=(3,3))(c2)

#3

c3=Conv2D(256 , (5,5) , activation="relu" , padding="same")(p2)
c3=Conv2D(256 , (5,5) , activation="relu" , padding="same")(c3)
c3=Conv2D(256 , (5,5) , activation="relu" , padding="same")(c3)
p3=MaxPooling2D(pool_size=(2,2))(c3)

#4

c4=Conv2D(256 , (5,5) , activation="relu" , padding="same")(p3)
c4=Conv2D(256 , (5,5) , activation="relu" , padding="same")(c4)
c4=Conv2D(256 , (5,5) , activation="relu" , padding="same")(c4)
p4=MaxPooling2D(pool_size=(2,2))(c4)

#5

c5=Conv2D(256 , (5,5) , activation="relu" , padding="same")(p4)
c5=Conv2D(256 , (5,5) , activation="relu" , padding="same")(c5)
c5=Conv2D(256 , (5,5) , activation="relu" , padding="same")(c5)
p5=MaxPooling2D(pool_size=(2,2))(c5)

#6

c6=Conv2D(256 , (5,5) , activation="relu" , padding="same")(p5)
c6=Conv2D(256 , (5,5) , activation="relu" , padding="same")(c6)
c6=Conv2D(256 , (5,5) , activation="relu" , padding="same")(c6)
p6=MaxPooling2D(pool_size=(2,2))(c6)

#fully connected layers
f=Flatten()(p3)
#FC1
fc1=Dense(1024)(f)
b1=BatchNormalization()(fc1)
ac=Activation("relu")(b1)
d1=Dropout(0.2)(ac)

#FC2
fc2=Dense(1024)(d1)
b2=BatchNormalization()(fc2)
ac=Activation("relu")(b2)
d2=Dropout(0.2)(ac)

#FC3
fc3=Dense(1024)(d2)
b2=BatchNormalization()(fc3)
ac=Activation("relu")(b2)
d3=Dropout(0.2)(ac)


#FC4
x=Dense(512 , activation="relu")(d3)
x=Dense(512 , activation="relu")(x)
x=Dense(512 , activation="relu")(x)
x=Dense(256 , activation="relu")(x)
x=Dense(128 , activation="relu")(x)

output=Dense(len(classes) , activation="sigmoid")(x)

model=Model(inputs=Inputs , outputs=output)

# Add a checkbox to show/hide the model summary
show_model_summary = st.checkbox("Show Model Summary")

# Display the model summary if the checkbox is checked
if show_model_summary:
    st.write("Model Summary:")
    model.summary()

# Add a checkbox to show/hide the training information
show_training_info = st.checkbox("Show Training Information")

# Compile
model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.binary_crossentropy,
              metrics=["accuracy"])

# Display the training information if the checkbox is checked
if show_training_info:
    st.write("Training Information:")


    X_train_dummy, _, y_train_dummy, _ = train_test_split(scans,Labels, test_size=0.2)
    X_val_dummy, _, y_val_dummy, _ = train_test_split(X_train_dummy, y_train_dummy, test_size=0.2)

    # Fit the model (replace with your actual data)
    model.fit(X_train_dummy, y_train_dummy, epochs=10, batch_size=32, validation_data=(X_val_dummy, y_val_dummy),
              verbose=1)

# Add a checkbox to show/hide the model summary and plot
show_model_info = st.checkbox("Show Model Information")

# Display the model summary and plot if the checkbox is checked
if show_model_info:
    st.write("Model Information:")

    # Fine-tuning dense net model model
    denenet_model = DenseNet169(weights="imagenet", include_top=False)

    for layer in denenet_model.layers[:150]:
        layer.trainable = False

    Inputs = Input((image_size, image_size, 3))
    c = Conv2D(512, (3, 3), activation="relu")(denenet_model(Inputs))
    c = Conv2D(512, (3, 3), activation="relu")(c)
    c = Conv2D(512, (3, 3), activation="relu", name="for_class_activation")(c)
    p = Flatten()(c)

    # Fine-tuning dense net model model
    x = Dense(2048, activation="relu")(p)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(1024, activation="relu")(x)

    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(1024, activation="relu")(x)

    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(512, activation="relu")(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)

    output = Dense(len(classes), activation="softmax")(x)

    model = Model(inputs=Inputs, outputs=output)

    # Display the model summary
    st.write("Model Summary:")
    model.summary()

    # Save the plot of the model
    plot_model(model, show_shapes=True, show_layer_names=True, to_file="model.png")
    st.image("model.png", use_column_width=True)

# Callbacks
    callbacks_denseNet = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint("NIH_model_2.h5", save_best_only=True, verbose=1),
    ]

    # Fit the model
    DenseNet_history = model.fit(train_generator,
                                 epochs=20,
                                 steps_per_epoch=X_train.shape[0]/batch_size,
                                 validation_data=val_generator,
                                 callbacks=callbacks_denseNet,
                                 verbose=1)

# Add a checkbox to show/hide the evaluation information
show_evaluation_info = st.checkbox("Show Evaluation Information")

# Display the evaluation information if the checkbox is checked
if show_evaluation_info:
    st.write("Evaluation Information:")
    model.evaluate(train_generator), model.evaluate(val_generator), model.evaluate(test_generator)

# Add checkboxes to show/hide the plots
show_accuracy_plot = st.checkbox("Show Accuracy Plot")
show_loss_plot = st.checkbox("Show Loss Plot")

# Display the plots if the checkboxes are checked
if show_accuracy_plot or show_loss_plot:
    st.write("- the Accuracy and Loss for DenseNet Model With 20 Epochs")
    plt.figure(figsize=(40, 20))

    # Summarize history for accuracy
    if show_accuracy_plot:
        plt.subplot(5, 5, 1)
        plt.plot(DenseNet_history.history['accuracy'])
        plt.plot(DenseNet_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

    # Summarize history for loss
    if show_loss_plot:
        plt.subplot(5, 5, 2)
        plt.plot(DenseNet_history.history['loss'])
        plt.plot(DenseNet_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'loss'], loc='upper left')

    # Show the plots in Streamlit
    st.pyplot(plt)




