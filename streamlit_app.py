import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import json

# Set page config
st.set_page_config(
    page_title="Traffic Sign Classifier", page_icon="🚦", layout="centered"
)


# Load the model (cache it to avoid reloading)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("traffic_sign_model_multilabel.h5")
    return model


# Load class names from JSON file
@st.cache_data
def load_class_names(file_path="classes.json"):
    try:
        with open(file_path, "r") as f:
            class_data = json.load(f)
            # Ensure classes are ordered correctly
            return [class_data[str(i)] for i in range(len(class_data))]
    except FileNotFoundError:
        st.error(f"Class file {file_path} not found!")
        return []
    except Exception as e:
        st.error(f"Error loading classes: {str(e)}")
        return []


model = load_model()
CLASS_NAMES = load_class_names()


# Image preprocessing function
def preprocess_image(image):
    # Resize to match model's expected sizing
    image = Image.fromarray(image)
    image = image.resize((30, 30))
    image = np.array(image)
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # Normalize pixel values
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image


# Function to resize displayed image while maintaining aspect ratio
def resize_display_image(image, max_width=500):
    img = Image.fromarray(image)
    width_percent = max_width / float(img.size[0])
    height = int((float(img.size[1]) * float(width_percent)))
    return img.resize((max_width, height), Image.LANCZOS)


# Main app function
def main():
    st.title("🚦 Traffic Sign Classifier")
    st.write("Upload an image of a traffic sign to classify it")

    if not CLASS_NAMES:
        st.error("Failed to load class names. Check classes.json file.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a traffic sign image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read and display the uploaded image (resized for display)
        image = np.array(Image.open(uploaded_file))
        display_image = resize_display_image(image)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)

        # Get the class with highest probability
        max_prob_index = np.argmax(predictions)
        max_prob = predictions[0][max_prob_index]
        predicted_class = CLASS_NAMES[max_prob_index]

        # Display results
        st.subheader("Prediction Result")
        st.success(f"Most likely traffic sign: **{predicted_class}**")
        st.metric("Confidence", f"{max_prob:.2%}")
        st.progress(float(max_prob), text="Confidence level")

        # Show runner-up predictions
        with st.expander("Show other possible predictions"):
            top_k = 5
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            for i in top_indices:
                st.write(f"- {CLASS_NAMES[i]}: {predictions[0][i]:.2%}")


if __name__ == "__main__":
    main()
