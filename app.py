import os
from datetime import datetime
from typing import Literal

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field
from supabase import Client, create_client

# --- Page Config ---
st.set_page_config(
    page_title="Cloud Pneumonia Screening System",
    page_icon="🫁",
    layout="wide",
)

# --- Structured Output Schema for Gemini ---
class XRayScreeningResult(BaseModel):
    diagnosis: Literal["NORMAL", "PNEUMONIA"] = Field(
        description="Primary screening classification of the chest radiograph."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Confidence percentage between 0 and 100.",
    )
    triage_priority: Literal["ROUTINE", "HIGH PRIORITY REVIEW"] = Field(
        description="Clinical triage urgency based on suspected findings."
    )
    radiological_findings: str = Field(
        description="Concise 1-2 sentence clinical summary of lung fields, opacities, or clarity."
    )


# --- Cloud Clients Initialization ---
@st.cache_resource
def init_clients():
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    supabase_client: Client = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )
    return gemini_client, supabase_client


try:
    gemini_client, supabase = init_clients()
except Exception:
    st.error(
        "Missing or invalid cloud credentials in Streamlit Secrets."
    )
    st.stop()


# --- Helper Functions ---
def analyze_xray_with_gemini(image: Image.Image) -> XRayScreeningResult:
    prompt = (
        "You are an AI radiology screening assistant analyzing a frontal chest X-ray radiograph. "
        "Evaluate the lung fields for signs of pneumonia (such as focal consolidation, infiltrates, "
        "or pleural effusion) versus clear normal lung parenchyma. "
        "Provide a structured screening assessment."
    )
    candidate_models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]
    last_error = None
    for model_name in candidate_models:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=XRayScreeningResult,
                    temperature=0.1,
                ),
            )
            return XRayScreeningResult.model_validate_json(response.text)
        except Exception as err:
            last_error = err
            continue
    raise RuntimeError(f"All Gemini models busy: {last_error}")


def log_scan_to_supabase(
    patient_ref: str, image_source: str, result: XRayScreeningResult
):
    payload = {
        "patient_ref": patient_ref,
        "image_source": image_source,
        "diagnosis": result.diagnosis,
        "confidence_score": round(result.confidence_score, 2),
        "triage_priority": result.triage_priority,
        "findings": result.radiological_findings,
    }
    supabase.table("pneumonia_scan_logs").insert(payload).execute()


def fetch_recent_logs(limit: int = 15) -> pd.DataFrame:
    response = (
        supabase.table("pneumonia_scan_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    if response.data:
        return pd.DataFrame(response.data)
    return pd.DataFrame()


# --- UI Header ---
st.title("🫁 Cloud-Assisted Pneumonia Screening & Triage System")
st.caption(
    "Cloud Architecture: Streamlit Community Cloud (Frontend) • Google Gemini Vision API (AI Inference) • Supabase PostgreSQL (Cloud Audit Logging)"
)
st.divider()

# --- Sidebar: Input Selection ---
with st.sidebar:
    st.header("1. Radiograph Input")
    patient_ref = st.text_input(
        "Patient / Scan Reference ID",
        value=f"SCAN-{datetime.now().strftime('%H%M%S')}",
    )

    input_mode = st.radio(
        "Select Image Source:",
        ["Upload Chest X-Ray", "Demo Sample: Normal", "Demo Sample: Pneumonia"],
    )

    uploaded_file = None
    if input_mode == "Upload Chest X-Ray":
        uploaded_file = st.file_uploader(
            "Upload Frontal Chest X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )

    st.info(
        "**Disclaimer:** This cloud tool is a clinical decision-support prototype and does not replace certified medical diagnosis."
    )

# --- Load Selected Image ---
selected_image = None
source_label = ""

if input_mode == "Upload Chest X-Ray" and uploaded_file is not None:
    selected_image = Image.open(uploaded_file).convert("RGB")
    source_label = uploaded_file.name
elif input_mode == "Demo Sample: Normal":
    sample_path = os.path.join("sample_xrays", "normal_sample.jpeg")
    if os.path.exists(sample_path):
        selected_image = Image.open(sample_path).convert("RGB")
        source_label = "sample_normal.jpeg"
    else:
        st.warning("Add `sample_xrays/normal_sample.jpeg` to your repo to use this preset.")
elif input_mode == "Demo Sample: Pneumonia":
    sample_path = os.path.join("sample_xrays", "pneumonia_sample.jpeg")
    if os.path.exists(sample_path):
        selected_image = Image.open(sample_path).convert("RGB")
        source_label = "sample_pneumonia.jpeg"
    else:
        st.warning("Add `sample_xrays/pneumonia_sample.jpeg` to your repo to use this preset.")

# --- Main Workspace ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("Radiograph Preview")
    if selected_image is not None:
        st.image(selected_image, caption=f"Source: {source_label}", use_container_width=True)
    else:
        st.info("Upload a chest X-ray or choose a Demo Sample from the sidebar to begin.")

with col2:
    st.subheader("Cloud AI Inference & Triage")
    if selected_image is not None:
        if st.button("Run Cloud AI Screening", type="primary", use_container_width=True):
            with st.spinner("Sending radiograph to Gemini Cloud API & logging to Supabase..."):
                try:
                    result = analyze_xray_with_gemini(selected_image)
                    log_scan_to_supabase(patient_ref, source_label, result)
                    st.session_state["last_result"] = result
                    st.success("Inference complete & audit record committed to Supabase Cloud DB.")
                except Exception as exc:
                    st.error(f"Cloud pipeline error: {exc}")

    if "last_result" in st.session_state:
        res: XRayScreeningResult = st.session_state["last_result"]
        if res.diagnosis == "PNEUMONIA":
            st.error(f"### Prediction: {res.diagnosis} ({res.triage_priority})")
        else:
            st.success(f"### Prediction: {res.diagnosis} ({res.triage_priority})")

        m1, m2 = st.columns(2)
        m1.metric("Model Confidence", f"{res.confidence_score:.1f}%")
        m2.metric("Triage Action", res.triage_priority)

        st.markdown("**AI Radiological Observations:**")
        st.write(res.radiological_findings)

# --- Live Cloud Audit Log Section ---
st.divider()
st.subheader("🗄️ Live Cloud Audit Logs (Supabase PostgreSQL)")

if st.button("Refresh Cloud Logs"):
    st.rerun()

try:
    logs_df = fetch_recent_logs(15)
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No scans logged yet. Run an inference above to populate the cloud database.")
except Exception as exc:
    st.warning(f"Could not fetch Supabase logs: {exc}")
