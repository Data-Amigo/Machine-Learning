#!/usr/bin/env python
# coding: utf-8

# #   Malaria Cells Detection Using Neural Network
# 
# Malaria is a life threatening disease that is spread by mosquitoes. The disease mainly impacts tropical and subtropical 
# 
# regions. Glovally there are a total of 229 million cases and 409000 cases worldwide. This makes it crucial to for early 
# 
# detection systems for Plasmodium infected cells. This has brought an important discussion in the data realm on the use of 
# 
# neural networks to automate cell detection and classification. This would greatly enhance the fight against malaria as it would
# 
# increase the speed and accuracy of malaria diagnosis thus improving public health worldwide.
# 
# 
# #  Problem Statement
# 
# 
# Malaria is a significant health problem glovally and it impacts millions of people worldwide. The technological advancement in
# 
# the current era and the emergence of artificial intelligence has raised the important question on the creating a system that 
# 
# can timely and accurately diagnosis the disease in order to increase effectiveness in treatement and in the management of the 
# 
# disease. The traditional detection system involved manually checking examining the blood samples using microscopes in order to 
# 
# detect the Plasmodium infected cells. This was a labor intensive process that was often done by well trained and experienced 
# 
# tetechnicians. However, this posed a problem of human error and also limited resources since getting well trained and 
# 
# experienced Technicians was time consuming and difficult.This report aims to provide a comprehensive model that would provide a 
# 
# solution to this critical problem.
# 

# # Methodology 
# 
# The proposed methodology for the malaria cell detections system using neural networks involved multi step process. The 
# 
# methodolodgy involved loading the necessary python libraries suchs as Numpy,Pandas,Matplotlib, Scikit-learn, TensorFlow and 
# 
# TensorFlow Datasets which were imported in order to facilitate data preporcessing, model development and evaluation.The image 
# 
# data was preporcessed by resizing the images to consistent size then it was input into a neural network mode.The model 
# 
# framework was constructed using the Kera Sequential API and Tensorflow. The model consists of different techniques such as
# 
# convolutional, pooling,dense layers and droput with the last layer using the sigmoid activation function for the binary 
# 
# calssification. This model is vary effective in capturing the eminent featurs of the Plasmodium-infected cells and accurately
# 
# being able to differentiate them from healthy cells.

# ## Loading the Required Packages
# 
# The Packages were loaded into python environment using the code below:

# In[2]:


# Loading the tensorflow packages to be used in loading the data
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds


# ## Loading the Data into Python
# 
# The dataset used for the study below was obtained for Tensorflow dataset and was ingested using the tfds.load(). The train
# 
# split was used to retreive teh training data.The shuffle parameter was supposed to make sure that the files were shuffled
# 
# before being loaded.

# In[27]:


# Loading the Malaria data

malaria, info = tfds.load('malaria', split='train', shuffle_files=True, with_info=True)


# In[6]:


# getting the dataset info
info


# In[7]:


# Checking the two different classes of images that we have

print('Num classes: ' + str(info.features['label'].num_classes))
print("Class names: " + str(info.features['label'].names))


# ## Visualizing the Malaria Dataset
# 
# This section will focus on visualizing the malaria dataset. This will help in providing visual insights of the data.

# In[29]:


# Visualizing the dataset

malaria_vis=tfds.visualization.show_examples(malaria,info)


# ## Feature Extraction
# 
# The section conducts feature extraction of the data.This step ensures that the image data and their associated labels are 
# 
# properly stored in separate lists, preparing the data for the subsequent model training and evaluation stages. The feature 
# 
# extraction process is a crucial step in the overall methodology, as it transforms the raw dataset into a format that can be 
# 
# effectively utilized by the neural network model for the malaria cell detection task.

# In[31]:


# Feature Extraction
train_images = []
train_labels = []

for images in malaria:
    train_images.append(images['image'].numpy())
    train_labels.append(images['label'].numpy())


# In[32]:


import cv2

train_images = []
train_labels = []

for images in malaria:
    image = images['image'].numpy()
    image = cv2.resize(image, (224, 224))  # Resize the image to 224x224
    train_images.append(image)
    train_labels.append(images['label'].numpy())

train_images = np.array(train_images)
train_labels = np.array(train_labels)


# In[35]:


print("Image:")
print(train_images[0])
print("Label: " + str(train_labels[0]))


# ## Building the Model
# 
# The data for the malaria cell detection model was loaded from the 'malaria' dataset provided by TensorFlow Datasets (TFDS). The 
# 
# dataset was split into training, validation, and test sets using the tfds.load() function with the split parameter. The 
# 
# training set consisted of 70% of the data, the validation set contained 15% of the data, and the remaining 15% was used for the 
# 
# test set. The shuffle_files=True parameter was used to ensure that the files were shuffled before being loaded, and the 
# 
# as_supervised=True parameter was set to indicate that the dataset contains both input features (images) and labels. The size of 
# 
# the input images was set to 200x200 pixels, and the batch size was set to 32. Finally, the number of images in each of the 
# 
# training, validation, and test sets was printed to provide an overview of the dataset used in the malaria cell detection model.

# In[36]:


# Building the model
BATCH_SIZE = 32
IMAGE_SIZE = [200, 200]

train_malaria, val_malaria, test_malaria = tfds.load('malaria',
                                      split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
                                      shuffle_files=True, as_supervised=True)


# In[37]:


NUM_TRAIN_IMAGES = tf.data.experimental.cardinality(train_malaria).numpy()
print("Num training images: " + str(NUM_TRAIN_IMAGES))

NUM_VAL_IMAGES = tf.data.experimental.cardinality(val_malaria).numpy()
print("Num validating images: " + str(NUM_VAL_IMAGES))

NUM_TEST_IMAGES = tf.data.experimental.cardinality(test_malaria).numpy()
print("Num testing images: " + str(NUM_TEST_IMAGES))


# In[38]:


for image, label in train_malaria.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


# ## Resizing Big Pictures
# 
# This section was resizing the big pictures in the dataset in order to improve the model perfomance by reducing the memory and
# 
# computational efficiency.

# In[14]:


# Resizing Big Pictures
def convert(image, label):
  image = tf.image.convert_image_dtype(image, tf.float32)
  return image, label

def pad(image,label):
  image,label = convert(image, label)
  image = tf.image.resize_with_crop_or_pad(image, 200, 200)
  return image,label


# In[39]:


padded_train_malaria = (
    train_malaria
    .cache()
    .map(pad)
    .batch(BATCH_SIZE)
) 

padded_val_malaria = (
    val_ds
    .cache()
    .map(pad)
    .batch(BATCH_SIZE)
) 


# In[40]:


image_batch, label_batch = next(iter(padded_train_malaria))

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title("uninfected")
        else:
            plt.title("parasitized")
        plt.axis("off")


# In[26]:


show_batch(image_batch.numpy(), label_batch.numpy())


# ## Building the Model
# 
# The model architecture for the malaria cell detection task was constructed using the TensorFlow Keras Sequential API. The input 
# 
# to the model is an image with dimensions specified by the IMAGE_SIZE variable, which is set to 180x180 pixels with 3 color 
# 
# channels. The model starts with two convolutional layers, each with 16 filters and a 3x3 kernel size, followed by a ReLU 
# 
# activation function and same padding to preserve the spatial dimensions. A max-pooling layer is then applied to downsample the 
# 
# feature maps. This is followed by a series of custom conv_block and dense_block functions, which encapsulate the convolutional 
# 
# and dense layers, respectively. The convolutional blocks progressively increase the number of filters, while the dense blocks 
# 
# reduce the dimensionality of the features and apply dropout to prevent overfitting. Finally, a single dense layer with a 
# 
# sigmoid activation function is used for the binary classification of malaria-infected and healthy cells. The model is compiled 
# 
# with the Adam optimizer and binary cross-entropy loss, with the Area Under the Curve (AUC) metric used to evaluate the model's 
# 
# performance.

# In[17]:


# Building the Model

def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
    
    return block

def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block


# In[18]:


def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model


# In[19]:


model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)


# In[20]:


# Defining Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("malaria_model.keras",
                                                  save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                    restore_best_weights=True)

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)


# ## Training the Model
# 
# The neural network model for malaria cell detection was trained using the model.fit() function, which takes the training 
# 
# dataset (padded_train_ds) and the validation dataset (padded_val_ds) as inputs. The training was conducted for 20 epochs, with 
# 
# the progress displayed in the console. During each epoch, the model's performance was evaluated on both the training and 
# 
# validation sets, with the Area Under the Curve (AUC) and loss metrics being reported. The training progress shows that the 
# 
# model's performance improved over the course of the 20 epochs, with the training AUC increasing from 0.42 to 0.79 and the 
# 
# training loss decreasing from 1.13 to 0.53. However, the validation metrics remained relatively stable, with the validation AUC 
# 
# staying around 0.50 and the validation loss fluctuating between 0.86 and 1.00. This suggests that the model may be overfitting 
# 
# to the training data, and additional techniques, such as regularization or data augmentation, may be necessary to improve its 
# 
# generalization capabilities.

# In[22]:


# Training the model
history = model.fit(
    padded_train_malaria, epochs=20,
    validation_data=padded_val_malaria,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]
)


# ## Model Perfomance
# 
# After training the model, its performance was evaluated on the test dataset. The test dataset was first preprocessed by padding 
# 
# the images to the required size, and then the model was evaluated using the model.evaluate() function. The evaluation results 
# 
# showed that the model achieved an AUC (Area Under the Curve) score of 0.5 and a loss of 0.8262 on the test set. This indicates 
# 
# that the model's performance on the unseen test data was not particularly strong, suggesting that the model may be overfitting 
# 
# to the training data or that the training dataset was not diverse enough to generalize well.

# In[23]:


# Evaluating Model Perfomance
padded_test_malaria = (
     test_malaria
    .cache()
    .map(pad)
    .batch(BATCH_SIZE)
) 


# In[24]:


model.evaluate(padded_test_malaria)


# In[25]:


model.summary()


# ## Conclusion
# 
# The malaria cell detection model developed in this study consists of a complex architecture with multiple convolutional, 
# 
# pooling, and dense layers. While the model was able to achieve reasonable performance on the training and validation datasets, 
# 
# its evaluation on the test set reveals limitations in its ability to generalize to new, unseen data. To improve the model's 
# 
# performance, further investigation is needed to address potential issues such as overfitting, data imbalance, or the need for 
# 
# more robust feature extraction techniques. Exploring alternative model architectures, incorporating advanced regularization 
# 
# methods, and expanding the diversity of the training dataset may help enhance the model's overall accuracy and real-world 
# 
# applicability for the malaria cell detection task.
