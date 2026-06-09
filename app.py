"""app.py — Intelligent Doctor Assistant Flask Web Application"""

import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
from agents.ida_orchestrator import IDAOrchestrator
from utils.helpers import load_config, get_logger

logger = get_logger("app")
cfg = load_config()
app = Flask(__name__)
app.secret_key = cfg["app"]["secret_key"]
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

ida = IDAOrchestrator(cfg)
patient_store  = {}
results_store  = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def analyze_uploaded_xray(file_storage) -> dict:
    """Save uploaded X-ray and run CheXzero analysis. Returns findings dict."""
    import tempfile
    suffix = Path(file_storage.filename).suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name
    try:
        from tools.xray_processor import analyze_xray_file
        result = analyze_xray_file(tmp_path, model_name=cfg["knowledge_graph"]["xray_model"])
        result["image_path"] = tmp_path
        return result
    except Exception as e:
        logger.error(f"X-ray analysis failed: {e}")
        return {"findings": [], "primary_finding": "Unknown",
                "normal": False, "error": str(e), "image_path": tmp_path}


def analyze_xray_base64(b64: str) -> dict:
    """Analyze X-ray from base64 string (from JSON API)."""
    try:
        from tools.xray_processor import analyze_xray_base64 as _analyze
        return _analyze(b64)
    except Exception as e:
        logger.error(f"X-ray base64 analysis failed: {e}")
        return {"findings": [], "primary_finding": "Unknown", "normal": False, "error": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "agents": ["IDA","SPA","DA","EA"]})


@app.route("/api/analyze-xray", methods=["POST"])
def analyze_xray_endpoint():
    """
    Dedicated endpoint: upload X-ray → get findings immediately.
    Accepts multipart/form-data with field 'xray_image'.
    Returns detected pathologies before full diagnosis.
    """
    if "xray_image" not in request.files:
        return jsonify({"error": "No xray_image file uploaded"}), 400

    file = request.files["xray_image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    allowed = {".dcm", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    if Path(file.filename).suffix.lower() not in allowed:
        return jsonify({"error": f"File type not supported. Use: {allowed}"}), 400

    result = analyze_uploaded_xray(file)
    return jsonify(result)


@app.route("/api/process", methods=["POST"])
def process():
    """
    Main endpoint. Accepts multipart/form-data OR JSON.

    multipart/form-data fields:
      xray_image   — X-ray file (.dcm/.png/.jpg)   ← MAIN INPUT
      patient_id   — string (optional, auto-generated)
      action       — "kg_only" | "predict" | "explain"
      age          — int
      gender       — "M" | "F"
      heart_rate   — int (optional)
      bp_systolic  — int (optional)
      bp_diastolic — int (optional)
      temperature  — float (optional)
      o2_saturation— int (optional)
      respiratory_rate — int (optional)
      symptoms     — comma-separated string (optional)
      clinical_narrative — string (optional)

    OR JSON body (same fields, xray_image as base64 string under key "xray_b64")
    """
    # ── Parse request ──────────────────────────────────────────────────────────
    is_multipart = request.content_type and "multipart" in request.content_type
    xray_result = {}

    if is_multipart:
        form = request.form
        action = form.get("action", "explain")
        patient_id = form.get("patient_id") or f"P{len(patient_store)+1:05d}"

        # Analyze uploaded X-ray image
        if "xray_image" in request.files and request.files["xray_image"].filename:
            logger.info("Analyzing uploaded X-ray image...")
            xray_result = analyze_uploaded_xray(request.files["xray_image"])
            logger.info(f"X-ray findings: {[f['pathology'] for f in xray_result.get('findings',[])]}")
        else:
            logger.warning("No X-ray image uploaded — proceeding with vitals only")

        def fget(k, default=None, cast=str):
            v = form.get(k, default)
            try: return cast(v) if v is not None else default
            except: return default

        symptoms_raw = fget("symptoms", "")
        symptoms = [s.strip() for s in symptoms_raw.split(",") if s.strip()] if symptoms_raw else []

        data = {
            "patient_id":        patient_id,
            "age":               fget("age",  60,   int),
            "gender":            fget("gender", "M"),
            "heart_rate":        fget("heart_rate",  80,   int),
            "bp_systolic":       fget("bp_systolic", 120,  int),
            "bp_diastolic":      fget("bp_diastolic", 80,  int),
            "temperature":       fget("temperature",  37.0, float),
            "o2_saturation":     fget("o2_saturation", 97,  int),
            "respiratory_rate":  fget("respiratory_rate", 16, int),
            "symptoms":          symptoms,
            "clinical_narrative": fget("clinical_narrative", ""),
        }
    else:
        # JSON request
        data = request.get_json() or {}
        action = data.pop("action", "explain")
        patient_id = data.get("patient_id") or f"P{len(patient_store)+1:05d}"
        data["patient_id"] = patient_id

        # X-ray from base64
        if "xray_b64" in data:
            logger.info("Analyzing base64 X-ray image...")
            xray_result = analyze_xray_base64(data.pop("xray_b64"))
            logger.info(f"X-ray findings: {[f['pathology'] for f in xray_result.get('findings',[])]}")

        if isinstance(data.get("symptoms"), str):
            data["symptoms"] = [s.strip() for s in data["symptoms"].split(",") if s.strip()]

    # ── Merge X-ray findings into patient record ───────────────────────────────
    # disease_labels come from X-ray analysis, NOT manual input
    if xray_result.get("findings"):
        data["disease_labels"] = [f["pathology"] for f in xray_result["findings"]]
        data["xray_confidence"] = {f["pathology"]: f["confidence"] for f in xray_result["findings"]}
    elif not data.get("disease_labels"):
        data["disease_labels"] = []

    data["image_path"] = xray_result.get("image_path", "")

    # ── Auto-generate clinical narrative if not provided ──────────────────────
    if not data.get("clinical_narrative"):
        syms = data.get("symptoms", [])
        data["clinical_narrative"] = (
            f"{data.get('age','?')}-year-old "
            f"{'male' if data.get('gender')=='M' else 'female'} presenting with "
            f"{', '.join(syms) if syms else 'presenting complaints'}. "
            f"Vitals: HR {data.get('heart_rate','?')} bpm, "
            f"BP {data.get('bp_systolic','?')}/{data.get('bp_diastolic','?')} mmHg, "
            f"Temp {data.get('temperature','?')}°C, "
            f"O2 {data.get('o2_saturation','?')}%."
        )

    patient_store[patient_id] = data

    try:
        result = ida.run(data, action=action)
        # Attach X-ray analysis metadata to response
        if xray_result:
            result["xray_analysis"] = {
                "findings": xray_result.get("findings", []),
                "primary_finding": xray_result.get("primary_finding", ""),
                "normal": xray_result.get("normal", False),
                "note": xray_result.get("note", ""),
            }
        results_store[patient_id] = result
        return jsonify(result)
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": str(e), "patient_id": patient_id}), 500


@app.route("/api/demo", methods=["POST"])
def demo():
    """Built-in demo patients (no X-ray upload needed for demo)."""
    body = request.get_json() or {}
    scenario = body.get("scenario", "chf")
    action   = body.get("action", "explain")

    patients = {
        "chf": {
            "patient_id": "DEMO_CHF_001", "age": 72, "gender": "M",
            "heart_rate": 108, "bp_systolic": 145, "bp_diastolic": 92,
            "temperature": 37.1, "o2_saturation": 89, "respiratory_rate": 22,
            "symptoms": ["dyspnea on exertion","bilateral leg edema","orthopnea"],
            "disease_labels": ["Cardiomegaly","Edema","Effusion"],
            "clinical_narrative": "72-year-old male with worsening dyspnea and bilateral leg edema.",
            "image_path": "",
        },
        "pneumonia": {
            "patient_id": "DEMO_PNE_001", "age": 45, "gender": "F",
            "heart_rate": 98, "bp_systolic": 118, "bp_diastolic": 76,
            "temperature": 38.9, "o2_saturation": 93, "respiratory_rate": 20,
            "symptoms": ["productive cough","fever","pleuritic chest pain"],
            "disease_labels": ["Pneumonia","Consolidation"],
            "clinical_narrative": "45-year-old female with 4-day fever and productive cough.",
            "image_path": "",
        },
        "normal": {
            "patient_id": "DEMO_NORM_001", "age": 34, "gender": "M",
            "heart_rate": 72, "bp_systolic": 118, "bp_diastolic": 76,
            "temperature": 36.7, "o2_saturation": 98, "respiratory_rate": 14,
            "symptoms": ["routine checkup"],
            "disease_labels": ["No Finding"],
            "clinical_narrative": "34-year-old male for routine checkup.",
            "image_path": "",
        },
    }

    patient = patients.get(scenario, patients["chf"])
    try:
        result = ida.run(patient, action=action)
        results_store[patient["patient_id"]] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/patient/<patient_id>/kg")
def get_kg(patient_id):
    return jsonify(ida.spa.get_kg_json(patient_id))


@app.route("/api/patient/<patient_id>/result")
def get_result(patient_id):
    return jsonify(results_store.get(patient_id, {}))


if __name__ == "__main__":
    app.run(host=cfg["app"]["host"], port=cfg["app"]["port"], debug=cfg["app"]["debug"])
