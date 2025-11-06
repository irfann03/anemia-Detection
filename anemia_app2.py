import streamlit as st
import cv2
import numpy as np
from PIL import Image

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
    st.title("Anemia Detection from Retinal Images")
    st.markdown("Upload a fundus image to check if it indicates anemia.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image using PIL and convert it to a NumPy array
        img = Image.open(uploaded_file)
        img_np = np.array(img)

        # Preprocess image (apply CLAHE on the red channel)
        preprocessed_img = preprocessing_for_model(img_np)

        # Display original image
        st.image(img, caption="Original Image", width=350)

        # Convert processed image to uint8 for proper display in Streamlit
        processed_img = convert_to_uint8(preprocessed_img)

        # Convert back to a PIL Image for Streamlit display
        processed_img_pil = Image.fromarray(processed_img)

        # Display processed image after CLAHE (if you want to compare it)
        st.image(processed_img_pil, caption="Processed Image with CLAHE", width=350)

        # Placeholder prediction (replace later with actual model prediction)
        st.write("**Prediction:** (Model not loaded yet)")

if __name__ == "__main__":
    main()
