import argparse
import sys
import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Config
IMG_SIZE = (224, 224)
MODEL_PATH = "../native-vs-recaptured-classifier/best_model_finetuned.weights.h5"


def load_model():
    """Build and load the MobileNetV2 model with trained weights."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    # Build model exactly as in training
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Load trained weights
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print(f"Model weights loaded from: {MODEL_PATH}")
    else:
        print(f"Error: Weights file not found at: {MODEL_PATH}")
        sys.exit(1)

    return model


def preprocess_image(image_path):
    """Preprocess image for prediction."""
    img = Image.open(image_path)
    img = img.resize(IMG_SIZE).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main():
    parser = argparse.ArgumentParser(description='Predict screen recapture probability')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='Path to model weights')
    args = parser.parse_args()
    
    # Update model path if provided
    global MODEL_PATH
    MODEL_PATH = args.model
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    model = load_model()
    img_array = preprocess_image(args.image_path)
    
    # Make prediction
    prob_recaptured = model.predict(img_array, verbose=0)[0][0]
    
    # Output: 0 = real photo, 1 = photo of screen
    print(prob_recaptured)


if __name__ == '__main__':
    main()
