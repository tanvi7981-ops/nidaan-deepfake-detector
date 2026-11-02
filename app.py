
import streamlit as st
import tempfile
import os
import torch
from transformers import pipeline

st.set_page_config(
    page_title="NIDAAN | Deepfake Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ NIDAAN")
st.subheader("AI-Powered Deepfake Video Detection")

st.write(
    "Upload a video and let our AI model analyze it."
)

st.warning(
    "Experimental result only. AI predictions can be wrong "
    "and should not be treated as proof."
)

MODEL_NAME = (
    "KoreaPeter/ms-eff-gcvit-deepfake-b0-ff-plus-plus"
)

@st.cache_resource
def load_detector():
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "video-classification",
        model=MODEL_NAME,
        trust_remote_code=True,
        device=device
    )

video = st.file_uploader(
    "Upload your video",
    type=["mp4", "avi", "mov", "mkv"]
)

if video:
    st.video(video)

    if st.button("Analyze Video", type="primary"):

        path = None

        try:
            suffix = os.path.splitext(video.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as f:
                f.write(video.getbuffer())
                path = f.name

            with st.spinner("Analyzing video..."):
                detector = load_detector()
                result = detector(
                    path,
                    num_frames=20,
                    return_frame_scores=True
                )

            st.subheader("Analysis Result")
            st.write(result)

            st.caption(
                "Model output is not a guaranteed probability "
                "or proof of authenticity."
            )

        except Exception as e:
            st.error("Video analysis failed.")
            st.code(str(e))

        finally:
            if path and os.path.exists(path):
                os.remove(path)
