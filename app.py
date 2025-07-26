import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import requests
import os
from io import BytesIO
from streamlit_image_comparison import image_comparison

# --- App Configuration ---
st.set_page_config(
    page_title="Interactive Style Transfer",
    page_icon="🎨",
    layout="wide"
)

# --- Model Management & Helper Functions ---
MODELS = {
    "Candy": "candy-9.onnx",
    "Mosaic": "mosaic-9.onnx",
    "Rain Princess": "rain-princess-9.onnx",
    "Udnie": "udnie-9.onnx"
}
MODEL_URL_BASE = "https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model/"

@st.cache_resource
def download_model(model_name):
    url = MODEL_URL_BASE + model_name
    if not os.path.exists(model_name):
        with st.spinner(f"Downloading model: {model_name}... (This happens only once)"):
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(model_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                st.error(f"Error downloading model: {e}")
                return None
    return model_name

@st.cache_resource
def get_inference_session(model_path):
    try:
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        st.error(f"Error loading ONNX session: {e}")
        return None

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    prep = transforms.Compose([
        transforms.Resize(480),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    return prep(img).unsqueeze(0).numpy().astype(np.float32), img

def run_style_transfer(session, input_tensor):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]
    final_output = np.clip(output[0], 0, 255).transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(final_output)

# --- Pre-download all models on app startup ---
for model_file in MODELS.values():
    download_model(model_file)

# --- UI Setup ---
st.title("🎨 Interactive Neural Style Transfer")

tab1, tab2 = st.tabs(["**🖼️ Style Transfer Tool**", "**📖 About the Project**"])

# --- Sidebar Controls ---
st.sidebar.header("CONTROLS")
uploaded_file = st.sidebar.file_uploader("1. Upload Your Image", type=["jpg", "jpeg", "png"])
selected_style = st.sidebar.selectbox("2. Select a Style", list(MODELS.keys()))
apply_button = st.sidebar.button("Apply Style", use_container_width=True, type="primary")

st.sidebar.write("---")
st.sidebar.markdown("Created by **Mohammad Sameer**")

# --- Tab 1: The Interactive Tool ---
with tab1:
    if uploaded_file is None:
        st.info("Upload an image using the sidebar to get started.", icon="👈")
    else:
        content_tensor, original_image = preprocess_image(uploaded_file)
        
        if apply_button:
            model_path = MODELS[selected_style]
            session = get_inference_session(model_path)
            
            if session:
                with st.spinner(f"Applying '{selected_style}' style..."):
                    stylized_image = run_style_transfer(session, content_tensor)

                st.markdown("### Drag the slider to see the difference!")
                image_comparison(
                    img1=original_image,
                    img2=stylized_image,
                    label1="Original",
                    label2=f"Stylized ({selected_style})",
                    width=800,
                    starting_position=50,
                    show_labels=True,
                )

                st.sidebar.write("---")
                st.sidebar.subheader("Download Image")
                
                download_format = st.sidebar.radio(
                    "Select format",
                    ("PNG", "JPEG"),
                    horizontal=True
                )
                
                buf = BytesIO()
                if download_format == "PNG":
                    stylized_image.save(buf, format="PNG")
                    mime_type = "image/png"
                elif download_format == "JPEG":
                    stylized_image.save(buf, format="JPEG")
                    mime_type = "image/jpeg"
                byte_im = buf.getvalue()

                st.sidebar.download_button(
                    label=f"Download as {download_format}",
                    data=byte_im,
                    file_name=f"stylized_{selected_style.lower()}.{download_format.lower()}",
                    mime=mime_type,
                    use_container_width=True
                )
            
        else:
            st.image(original_image, caption="Your Uploaded Image", use_column_width=True)
            st.warning("Select a style and click 'Apply Style' in the sidebar.", icon="👆")

# --- Tab 2: Project Information ---
with tab2:
    st.header("About Neural Style Transfer")
    st.markdown("""
    Neural Style Transfer is a fascinating Deep Learning technique that merges two images: a **content** image (like a photo you upload) and a **style** image (like a famous painting). The result is a new image where the content of your photo is "repainted" in the artistic style of the other image.
    """)
    st.subheader("How Does This App Work?")
    st.markdown("""
    This app uses a method called **Fast Neural Style Transfer** with pre-trained **ONNX models**. This allows for near-instant results because the "learning" of the style has already been done. The app simply performs a single, rapid **inference** pass to create your new image.
    """)
    st.subheader("Technology Stack")
    st.markdown("""
    - **App Framework:** [Streamlit](https://streamlit.io/)
    - **Interactive Components:** [streamlit-image-comparison](https://github.com/fcakyon/streamlit-image-comparison)
    - **Model Format & Inference:** [ONNX Runtime](https://onnxruntime.ai/)
    - **Core ML/DL Libraries:** [PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/)
    - **Image Handling:** [Pillow](https://python-pillow.org/)
    """)