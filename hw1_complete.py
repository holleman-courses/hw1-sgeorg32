#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


##

def build_model1():
    model = tf.keras.Sequential([
        Input(shape=(32, 32, 3)),
        layers.Flatten(),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01)),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01)),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01)),
        layers.Dense(10)
    ])  # Add code to define model 1.
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def build_model2():
    model = tf.keras.Sequential([
        Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        +
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        #layers.Dense(128, activation=layers.LeakyReLU(alpha=0.01)),
        layers.Dense(10)
    ])  # Add code to define model 1.
    model.compile(
        optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def build_model3():
    inputs = Input(shape=(32, 32, 3))
    x = layers.SeparableConv2D(32, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.SeparableConv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    for _ in range(4):
        x = layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(10)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    ## This one should use the functional API so you can create the residual connections
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def build_model50k():
    model = tf.keras.Sequential([
        Input(shape=(32, 32, 3)),
        layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':
    ########################################
    ## Add code here to Load the CIFAR10 data set
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    val_frac = 0.1
    num_val_samples = int(len(train_images) * val_frac)
    val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
    trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)

    val_images = train_images[val_idxs]
    val_labels = train_labels[val_idxs]
    train_images = train_images[trn_idxs]
    train_labels = train_labels[trn_idxs]

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()
    val_labels = val_labels.squeeze()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    val_images = val_images / 255.0

    ## Build and train model 1
    model1 = build_model1()
    model1.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
    # compile and train model 1.

    ## Build, compile, and train model 2 (DS Convolutions)
    model2 = build_model2()
    model2.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))
    ### Repeat for model 3 and your best sub-50k params model
    model3 = build_model3()
    model3.fit(train_images, train_labels, epochs=30, validation_data=(val_images, val_labels))

    model50k = build_model50k()
    model50k.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
    model50k.save("best_model.h5")

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


    test_img = np.array(keras.utils.load_img(
        './test_image_cat.jpg',
         grayscale=False,
        color_mode='rgb',
        target_size=(32, 32)))

    test_img_input = np.expand_dims(test_img / 255.0, axis=0)

    predictions = model50k.predict(test_img_input)
    predicted_class = class_names[np.argmax(predictions)]
    print(f"Predicted Class: {predicted_class}")
