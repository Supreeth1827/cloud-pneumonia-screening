# 🫁 Cloud-Assisted Automated Pneumonia Screening & Triage System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cloud-pneumonia-screening-n9gfgeezkxu6ly2sfnqtmf.streamlit.app)

A cloud-native clinical decision-support prototype that performs real-time pneumonia screening and triage from frontal chest X-ray radiographs, persisting timestamped audit logs in a managed cloud PostgreSQL database.

🔗 **Live Cloud App:** [https://cloud-pneumonia-screening-n9gfgeezkxu6ly2sfnqtmf.streamlit.app](https://cloud-pneumonia-screening-n9gfgeezkxu6ly2sfnqtmf.streamlit.app)

---

## 🏗️ Cloud Architecture

The application decouples presentation, multimodal AI inference, and relational persistence across three cloud services:

1. **Frontend & Application Hosting — Streamlit Community Cloud**
   - Continuously deployed from GitHub with secure cloud secret management.
   - Preprocesses radiographs in RGB and downscales to `768×768` pixels for low-latency cloud transit (~2–3 seconds end-to-end).
2. **Cloud AI Inference — Google Gemini Vision API**
   - Uses `gemini-3.5-flash` (with automatic multi-model fallback to `gemini-3.5-flash-lite` and `gemini-3.8-flash`).
   - Enforces a structured Pydantic JSON schema (`XRayScreeningResult`) returning the diagnostic verdict (`NORMAL` or `PNEUMONIA`), confidence score, triage priority (`ROUTINE` or `HIGH PRIORITY REVIEW`), and radiological observations.
3. **Cloud Database & Audit Trail — Supabase Cloud PostgreSQL**
   - Logs every scan record into `public.pneumonia_scan_logs` with Row-Level Security (RLS) enabled and displays live audit history directly on the dashboard.

---

## ⚙️ Workflow

1. **Radiograph Input:** Enter a Patient / Scan Reference ID and upload a frontal chest X-ray (`.jpg`, `.jpeg`, `.png`) or select a built-in demo sample (`Normal` or `Pneumonia`).
2. **Payload Optimization:** Converts the radiograph to RGB and scales it to `768×768` px.
3. **Cloud Vision Inference:** Sends the image and clinical screening prompt to the Google Gemini Vision API with structured JSON output enforcement.
4. **Cloud Audit Commit:** Persists the scan metadata, classification, confidence percentage, and findings into Supabase PostgreSQL.
5. **Triage Dashboard:** Displays a color-coded priority alert (`ROUTINE` vs. `HIGH PRIORITY REVIEW`) alongside live cloud audit logs.

---

## 📂 Repository Structure

```text
cloud-pneumonia-screening/
├── app.py                  # Main Streamlit application, Gemini API client & Supabase logger
├── requirements.txt        # Python dependencies (streamlit, google-genai, supabase, pydantic, Pillow)
├── sample_xrays/           # Built-in demo radiographs for one-click validation
│   ├── normal_sample.jpeg
│   └── pneumonia_sample.jpeg
└── README.md               # Project documentation
