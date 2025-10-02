import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import io

st.title("MNIST-Style 28x28 Digit Drawing Canvas")
st.write("Draw digits like in the MNIST dataset - white strokes on black background")

# Canvas settings for MNIST-style
canvas_size = 280  # 28 * 10 for better drawing experience

# Create the drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
    stroke_width=st.slider("Brush size", 5, 30, 15),  # Thicker brush for digit drawing
    stroke_color="white",  # White strokes like MNIST
    background_color="black",  # Black background like MNIST
    height=canvas_size,
    width=canvas_size,
    drawing_mode="freedraw",
    key="mnist_canvas",
)

# Process the drawing
if canvas_result.image_data is not None:
    # Convert canvas to proper MNIST format
    img_data = canvas_result.image_data
    
    # Convert RGBA to RGB
    img_rgb = Image.fromarray(img_data.astype('uint8'), 'RGBA')
    img_rgb = img_rgb.convert('RGB')
    
    # Resize to 28x28
    img_28x28 = img_rgb.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to grayscale
    img_gray = img_28x28.convert('L')
    
    # Convert to numpy array and normalize like MNIST
    mnist_array = np.array(img_gray)
    
    # MNIST normalization (0-255 range, inverted so background is 0)
    mnist_normalized = mnist_array.astype(np.float32) / 255.0
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Your Drawing")
        st.image(canvas_result.image_data, width=200)
    
    with col2:
        st.subheader("28x28 MNIST Format")
        # Scale up for display with nearest neighbor to show pixels clearly
        display_img = img_gray.resize((200, 200), Image.Resampling.NEAREST)
        st.image(display_img, width=200)
    
    with col3:
        st.subheader("Statistics")
        st.write(f"**Shape:** {mnist_array.shape}")
        st.write(f"**Min pixel value:** {mnist_array.min()}")
        st.write(f"**Max pixel value:** {mnist_array.max()}")
        st.write(f"**Mean:** {mnist_array.mean():.2f}")
        st.write(f"**Non-zero pixels:** {np.count_nonzero(mnist_array)}")
    
    # Show pixel values option
    if st.expander("Show 28x28 pixel array"):
        st.write("Raw pixel values (0-255):")
        st.text(str(mnist_array))
        
        st.write("Normalized values (0.0-1.0):")
        st.text(str(np.round(mnist_normalized, 3)))
    
    # Save options
    st.subheader("Save Your Digit")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save as PNG"):
            img_bytes = io.BytesIO()
            img_gray.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            st.download_button(
                label="📁 Download PNG (28x28)",
                data=img_bytes,
                file_name="mnist_digit_28x28.png",
                mime="image/png"
            )
    
    with col2:
        if st.button("💾 Save as NumPy"):
            # Save as .npy file for ML use
            npy_bytes = io.BytesIO()
            np.save(npy_bytes, mnist_array)
            npy_bytes = npy_bytes.getvalue()
            
            st.download_button(
                label="📁 Download NumPy Array",
                data=npy_bytes,
                file_name="mnist_digit_28x28.npy",
                mime="application/octet-stream"
            )
    
    # Additional MNIST-style processing options
    st.subheader("MNIST Preprocessing Options")
    
    if st.checkbox("Apply center of mass centering"):
        # Calculate center of mass
        y_indices, x_indices = np.indices(mnist_array.shape)
        total_mass = mnist_array.sum()
        
        if total_mass > 0:
            center_y = (y_indices * mnist_array).sum() / total_mass
            center_x = (x_indices * mnist_array).sum() / total_mass
            
            st.write(f"Center of mass: ({center_x:.1f}, {center_y:.1f})")
            st.write("Note: MNIST digits are typically centered using center of mass")
    
    if st.checkbox("Show intensity histogram"):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(mnist_array.flatten(), bins=50, alpha=0.7, color='blue')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title('Pixel Intensity Distribution')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Clear button
if st.button("🗑️ Clear Canvas"):
    st.rerun()

# Instructions
with st.expander("📖 Instructions & MNIST Info"):
    st.markdown("""
    ### How to use:
    1. **Draw a digit (0-9)** on the black canvas using white strokes
    2. **Adjust brush size** using the slider above the canvas
    3. **Your drawing** will automatically be converted to 28x28 pixels
    4. **Download** as PNG image or NumPy array for machine learning
    
    ### MNIST Dataset Format:
    - **Size:** 28x28 pixels
    - **Colors:** Grayscale (0-255)
    - **Background:** Black (pixel value 0)
    - **Foreground:** White/Gray (pixel values > 0)
    - **Content:** Handwritten digits 0-9
    - **Normalization:** Often divided by 255 to get values between 0-1
    
    ### Tips for MNIST-like digits:
    - Draw **thick, clear digits** - thin lines may disappear when resized
    - **Center your digit** in the canvas
    - Make sure the digit **fills a good portion** of the canvas
    - **Avoid touching the edges** - leave some black border
    """)