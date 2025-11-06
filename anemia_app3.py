import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Load the model (ensure the correct file path is used)
model_path = "anemia_classifier_model.keras"  # Adjust if the model is in a different location
if not os.path.exists(model_path):
    st.error("Model file not found!")
else:
    model = load_model(model_path)
# CLAHE on red channel
def apply_clahe_to_red(img_array, mask=None, clip_limit=5.7, tile_size=(8, 8)):
    # Check if the image has an alpha channel (RGBA)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB

    b, g, r = cv2.split(img_array)  # Split the image into blue, green, and red channels
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    r_clahe = clahe.apply(r)  # Apply CLAHE to the red channel
    if mask is not None:
        r_final = r.copy()
        r_final[mask] = r_clahe[mask]  # Apply the CLAHE transformation to the red channel based on the mask
    else:
        r_final = r_clahe  # Otherwise, use the processed red channel without the mask
    result = cv2.merge([b, g, r_final])  # Merge the channels back together
    return result

# Mask for red threshold
def get_mask(img_array, threshold=20):
    # Check if the image has an alpha channel (RGBA)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB

    _, _, r = cv2.split(img_array)  # Extract the red channel
    mask = r > threshold  # Create a mask where red values are above the threshold
    return mask

# Preprocess image for model (resize only if needed for model, not for CLAHE)
def preprocessing_for_model(img):
    # No resizing here, keep original image size
    mask = get_mask(img)  # Get the red channel mask
    img_clahe = apply_clahe_to_red(img, mask)  # Apply CLAHE to the red channel
    img = img_clahe.astype(np.float32) / 255.0  # Normalize the image to the range [0, 1]
    return img

# Convert processed image back to uint8 for display
def convert_to_uint8(img):
    img = (img * 255).astype(np.uint8)
    return img

# Streamlit frontend
def main():
    st.title("Anemia Detection from Conjective Images")
    st.markdown("Upload a conjective image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image using PIL and convert it to a NumPy array
        img = Image.open(uploaded_file)
        img_np = np.array(img)

        # Preprocess image (apply CLAHE on the red channel)
        preprocessed_img = preprocessing_for_model(img_np)

        # Create columns for side-by-side layout
        col1, col2 = st.columns(2)

        # Display original image on the left (col1)
        with col1:
            st.image(img, caption="Original Image", width=350)

        # Convert processed image to uint8 for proper display in Streamlit
        processed_img = convert_to_uint8(preprocessed_img)

        # Convert back to a PIL Image for Streamlit display
        processed_img_pil = Image.fromarray(processed_img)

        # Display processed image after CLAHE (on the right side in col2)
        with col2:
            st.image(processed_img_pil, caption="Processed Image with CLAHE", width=350)


        # Prepare image for prediction
        resized_img = cv2.resize(preprocessed_img, (224, 224))
        input_img = np.expand_dims(resized_img, axis=0)

        # Make prediction
        prediction = model.predict(input_img)
        class_names = ['Anemic', 'Non-anemic']
        predicted_class = class_names[np.argmax(prediction)]

        # Placeholder prediction (replace later with actual model prediction)
        st.write(f"**Prediction:** {predicted_class}")

if __name__ == "__main__":
    main()
