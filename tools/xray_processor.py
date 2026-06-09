"""tools/xray_processor.py
Processes uploaded X-ray images using CheXzero-style CLIP zero-shot classification.
Supports DICOM (.dcm) and standard image formats (.png, .jpg, .jpeg).
Returns detected pathology findings automatically — user never types findings.
"""

import os
import sys
import base64
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.helpers import get_logger

logger = get_logger("xray_processor")

# NIH disease prompts for zero-shot CLIP classification
DISEASE_PROMPTS = {
    "Cardiomegaly":       ["enlarged heart on chest x-ray", "cardiomegaly chest radiograph", "increased cardiothoracic ratio"],
    "Edema":              ["pulmonary edema chest x-ray", "bilateral pulmonary edema", "interstitial edema chest"],
    "Effusion":           ["pleural effusion chest x-ray", "blunting costophrenic angle", "fluid in pleural space"],
    "Pneumonia":          ["pneumonia chest x-ray", "lobar consolidation pneumonia", "air space opacity pneumonia"],
    "Atelectasis":        ["atelectasis chest x-ray", "collapsed lung", "linear atelectasis"],
    "Consolidation":      ["pulmonary consolidation chest x-ray", "airspace consolidation", "lobar consolidation"],
    "Pneumothorax":       ["pneumothorax chest x-ray", "collapsed lung air pleural space", "absent lung markings"],
    "Infiltration":       ["pulmonary infiltrate chest x-ray", "lung infiltration", "haziness lung fields"],
    "Mass":               ["lung mass chest x-ray", "pulmonary mass lesion"],
    "Nodule":             ["pulmonary nodule chest x-ray", "lung nodule solitary"],
    "Emphysema":          ["emphysema chest x-ray", "hyperinflated lungs emphysema", "flattened diaphragm"],
    "Fibrosis":           ["pulmonary fibrosis chest x-ray", "interstitial fibrosis reticular pattern"],
    "Pleural_Thickening": ["pleural thickening chest x-ray", "thickened pleura"],
    "Hernia":             ["diaphragmatic hernia chest x-ray", "bowel loops chest"],
    "No Finding":         ["normal chest x-ray", "no acute cardiopulmonary findings", "clear lungs"],
}


def load_image_from_upload(file_path: str):
    """
    Load X-ray image from file path.
    Handles DICOM (.dcm) and standard formats (.png, .jpg, .jpeg).
    Returns a PIL Image in RGB mode.
    """
    from PIL import Image
    ext = Path(file_path).suffix.lower()

    if ext == ".dcm":
        try:
            import pydicom
            import numpy as np
            ds = pydicom.dcmread(file_path)
            arr = ds.pixel_array.astype(float)
            # Normalize to 0-255
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            arr = arr.astype("uint8")
            # Convert grayscale to RGB
            if arr.ndim == 2:
                img = Image.fromarray(arr).convert("RGB")
            else:
                img = Image.fromarray(arr[:, :, 0]).convert("RGB")
            return img
        except ImportError:
            logger.warning("pydicom not installed. Install with: pip install pydicom")
            raise
    else:
        return Image.open(file_path).convert("RGB")


def load_image_from_base64(b64_string: str, filename: str = "xray.png"):
    """Load X-ray image from base64 string (from web upload)."""
    from PIL import Image
    # Strip data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


class XrayAnalyzer:
    """
    Analyzes chest X-ray images using CLIP zero-shot classification (CheXzero style).
    Automatically extracts pathology findings — no manual input needed.
    Falls back to a lightweight rule-based analysis if CLIP unavailable.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            logger.info(f"Loading CLIP model: {self.model_name}")
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)
            self._model.eval()
            self._device = device
            logger.info(f"CLIP loaded on {device}")
        except Exception as e:
            logger.warning(f"CLIP not available ({e}). Using image-feature fallback.")
            self._model = None
        self._loaded = True

    def analyze(self, image, threshold: float = 0.08, top_k: int = 5) -> dict:
        """
        Run zero-shot classification on a chest X-ray PIL image.

        Returns:
        {
            "findings": [{"pathology": str, "confidence": float, "source": str}],
            "primary_finding": str,
            "normal": bool,
            "raw_scores": {disease: score}
        }
        """
        self._load()

        if self._model is not None:
            return self._clip_analyze(image, threshold, top_k)
        else:
            return self._feature_analyze(image)

    def _clip_analyze(self, image, threshold, top_k):
        import torch
        diseases = list(DISEASE_PROMPTS.keys())
        # Use first prompt for each disease (ensemble improves accuracy but costs more)
        texts = [DISEASE_PROMPTS[d][0] for d in diseases]

        inputs = self._processor(
            text=texts, images=image,
            return_tensors="pt", padding=True, truncation=True
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image[0].softmax(dim=0).cpu().numpy()

        raw_scores = {d: float(p) for d, p in zip(diseases, probs)}
        findings = []
        for disease, prob in raw_scores.items():
            if disease != "No Finding" and prob >= threshold:
                findings.append({
                    "pathology": disease,
                    "confidence": round(prob, 4),
                    "source": "CheXzero-CLIP"
                })
        findings.sort(key=lambda x: x["confidence"], reverse=True)
        findings = findings[:top_k]

        normal = len(findings) == 0 or raw_scores.get("No Finding", 0) > 0.4
        primary = findings[0]["pathology"] if findings else "No Finding"

        return {
            "findings": findings,
            "primary_finding": primary,
            "normal": normal,
            "raw_scores": raw_scores,
        }

    def _feature_analyze(self, image):
        """
        Lightweight fallback: uses image statistics (brightness, contrast, histogram)
        to give a rough indication when CLIP is not available.
        Not clinically reliable — for demo/dev only.
        """
        import numpy as np
        arr = np.array(image.convert("L")).astype(float) / 255.0

        mean_brightness = float(arr.mean())
        std_contrast    = float(arr.std())
        dark_ratio      = float((arr < 0.3).mean())   # lung fields = dark
        bright_ratio    = float((arr > 0.7).mean())   # bones/heart = bright
        asymmetry       = float(abs(arr[:, :arr.shape[1]//2].mean() - arr[:, arr.shape[1]//2:].mean()))

        findings = []
        raw_scores = {}

        # Heuristic rules (very rough)
        if bright_ratio > 0.25:
            findings.append({"pathology": "Cardiomegaly", "confidence": round(0.5 + bright_ratio, 3), "source": "feature-heuristic"})
        if dark_ratio < 0.35 and mean_brightness > 0.45:
            findings.append({"pathology": "Consolidation", "confidence": round(0.45 + std_contrast, 3), "source": "feature-heuristic"})
        if asymmetry > 0.06:
            findings.append({"pathology": "Effusion", "confidence": round(0.4 + asymmetry * 3, 3), "source": "feature-heuristic"})
        if std_contrast < 0.18:
            findings.append({"pathology": "Edema", "confidence": round(0.4 + (0.18 - std_contrast) * 5, 3), "source": "feature-heuristic"})

        for f in findings:
            raw_scores[f["pathology"]] = f["confidence"]

        if not findings:
            findings = [{"pathology": "No Finding", "confidence": 0.75, "source": "feature-heuristic"}]

        findings.sort(key=lambda x: x["confidence"], reverse=True)
        primary = findings[0]["pathology"] if findings[0]["pathology"] != "No Finding" else "No Finding"

        return {
            "findings": findings[:5],
            "primary_finding": primary,
            "normal": primary == "No Finding",
            "raw_scores": raw_scores,
            "note": "CLIP model not available — using image feature heuristics (dev mode only)"
        }


# Singleton
_analyzer = None

def get_analyzer(model_name: str = "openai/clip-vit-large-patch14") -> XrayAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = XrayAnalyzer(model_name)
    return _analyzer


def analyze_xray_file(file_path: str, model_name: str = "openai/clip-vit-large-patch14") -> dict:
    """Convenience function: load file and analyze."""
    image = load_image_from_upload(file_path)
    return get_analyzer(model_name).analyze(image)


def analyze_xray_base64(b64: str, filename: str = "xray.png") -> dict:
    """Convenience function: load base64 and analyze."""
    image = load_image_from_base64(b64, filename)
    return get_analyzer().analyze(image)
