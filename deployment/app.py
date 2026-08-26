import streamlit as st
import requests
import io
import base64
from PIL import Image

st.set_page_config(
    page_title="CDSS Skin Cancer Diagnostic Assistant", page_icon="🩺", layout="wide"
)

# Title & Clinical Disclaimer
st.title("🩺 Multimodal Clinical Decision Support System (CDSS-R)")
st.caption("Uncertainty-Aware Neural Network for Skin Cancer Risk Stratification")

st.warning(
    "**RESEARCH DISCLAIMER:** This system is an experimental decision support tool "
    "intended strictly for academic research and validation. It is NOT a certified clinical diagnostic device."
)

st.divider()

# Sidebar: Inputs
st.sidebar.header("📋 Patient Clinical Inputs")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dermoscopic Lesion Image", type=["jpg", "jpeg", "png"]
)

age = st.sidebar.slider("Patient Age", min_value=1, max_value=100, value=52)
clin_size = st.sidebar.number_input(
    "Lesion Diameter (mm)", min_value=0.5, max_value=50.0, value=8.5, step=0.5
)
sex = st.sidebar.selectbox("Biological Sex", ["Female", "Male"])
anatom_site = st.sidebar.selectbox(
    "Anatomical Site", ["Torso", "Head / Neck", "Extremities / Other"]
)

api_url = st.sidebar.text_input(
    "FastAPI Endpoint URL", value="http://127.0.0.1:8000/predict"
)

run_button = st.sidebar.button(
    "🔬 Run Multimodal Analysis", type="primary", use_container_width=True
)

# Main Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🖼️ Input Image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Dermoscopic Image", use_container_width=True)
    else:
        st.info(
            "Please upload a dermoscopic image in the sidebar to perform evaluation."
        )

with col2:
    st.subheader("📊 Diagnostic Predictions & Risk Triage")
    if run_button and uploaded_file is not None:
        with st.spinner(
            "Executing Inference, MC Dropout Sampling & Grad-CAM++ Attribution..."
        ):
            # Prepare multipart request
            uploaded_file.seek(0)
            files = {
                "image_file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            data = {
                "age": str(age),
                "clin_size_mm": str(clin_size),
                "sex_male": "1" if sex == "Male" else "0",
                "sex_female": "1" if sex == "Female" else "0",
                "site_torso": "1" if anatom_site == "Torso" else "0",
                "site_head_neck": "1" if anatom_site == "Head / Neck" else "0",
            }

            try:
                response = requests.post(api_url, files=files, data=data)
                if response.status_code == 200:
                    res = response.json()
                    pred = res["prediction"]
                    expl = res["explanations"]

                    # Display Metrics Metrics
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric(
                        "Malignancy Probability", f"{pred['probability'] * 100:.1f}%"
                    )
                    m_col2.metric(
                        "Epistemic Variance (σ²)", f"{pred['epistemic_variance']:.5f}"
                    )
                    m_col3.metric("Risk Classification", pred["risk_category"])

                    # Referral Alert Box
                    if pred["clinical_referral_flag"]:
                        st.error(
                            f"🚨 **REFERRAL RECOMMENDED:** {pred['triage_recommendation']}"
                        )
                    else:
                        st.success(
                            f"✅ **NORMAL CONFIDENCE:** {pred['triage_recommendation']}"
                        )

                    st.divider()
                    st.subheader("🔍 Visual Explanation (Grad-CAM++)")

                    # Render Overlay Heatmap
                    heatmap_bytes = base64.b64decode(expl["heatmap_overlay_base64"])
                    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                    st.image(
                        heatmap_img,
                        caption="Grad-CAM++ Visual Attention Heatmap",
                        use_container_width=True,
                    )

                    # Render Tabular Feature Importance
                    st.subheader("📋 Tabular Feature Contributions")
                    st.json(expl["tabular_feature_contributions"])

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(
                    f"Failed to connect to FastAPI endpoint at {api_url}. Is the server running? Error: {e}"
                )
