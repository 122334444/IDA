"""utils/helpers.py — shared utilities across all agents."""

import yaml
import logging
import os
from pathlib import Path


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── Logger ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Ontology Mapper ─────────────────────────────────────────────────────────

# Simplified ontology mappings (extend with real ontology IDs in production)
ONTOLOGY_MAP = {
    # SYMP - Symptom Ontology
    "dyspnea": {"class": "Symptom", "ontology": "SYMP", "id": "SYMP:0000305"},
    "shortness of breath": {"class": "Symptom", "ontology": "SYMP", "id": "SYMP:0000305"},
    "chest pain": {"class": "Symptom", "ontology": "SYMP", "id": "SYMP:0000318"},
    "fatigue": {"class": "Symptom", "ontology": "SYMP", "id": "SYMP:0000380"},
    "edema": {"class": "Symptom", "ontology": "SYMP", "id": "SYMP:0000197"},
    "fever": {"class": "Symptom", "ontology": "SYMP", "id": "SYMP:0000315"},
    "cough": {"class": "Symptom", "ontology": "SYMP", "id": "SYMP:0000614"},
    "nausea": {"class": "Symptom", "ontology": "SYMP", "id": "SYMP:0000249"},

    # DO - Disease Ontology
    "congestive heart failure": {"class": "Disease", "ontology": "DO", "id": "DOID:6000"},
    "heart failure": {"class": "Disease", "ontology": "DO", "id": "DOID:6000"},
    "pneumonia": {"class": "Disease", "ontology": "DO", "id": "DOID:552"},
    "cardiomegaly": {"class": "Finding", "ontology": "RadLex", "id": "RID:5350"},
    "atelectasis": {"class": "Finding", "ontology": "RadLex", "id": "RID:28493"},
    "effusion": {"class": "Finding", "ontology": "RadLex", "id": "RID:28454"},
    "consolidation": {"class": "Finding", "ontology": "RadLex", "id": "RID:28545"},

    # Demographics
    "age": {"class": "Demographics", "ontology": "Demographics", "id": "DEMO:001"},
    "gender": {"class": "Demographics", "ontology": "Demographics", "id": "DEMO:002"},
    "heart rate": {"class": "Vital", "ontology": "FMA", "id": "FMA:9740"},
    "blood pressure": {"class": "Vital", "ontology": "FMA", "id": "FMA:9753"},
    "temperature": {"class": "Vital", "ontology": "FMA", "id": "FMA:20707"},
    "oxygen saturation": {"class": "Vital", "ontology": "FMA", "id": "FMA:9371"},
}


def map_entity_to_ontology(entity: str) -> dict:
    """Map extracted entity to its ontology class."""
    entity_lower = entity.lower().strip()
    # Exact match
    if entity_lower in ONTOLOGY_MAP:
        return {**ONTOLOGY_MAP[entity_lower], "text": entity}
    # Partial match
    for key, val in ONTOLOGY_MAP.items():
        if key in entity_lower or entity_lower in key:
            return {**val, "text": entity}
    # Default
    return {"class": "Unknown", "ontology": "Unknown", "id": "UNK:000", "text": entity}


# ─── NIH Disease Labels ───────────────────────────────────────────────────────

NIH_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia", "No Finding"
]

# Disease → clinical category mapping
DISEASE_CATEGORY = {
    "Cardiomegaly": "CHF",
    "Edema": "CHF",
    "Effusion": "CHF/Infection",
    "Pneumonia": "Respiratory Infection",
    "Atelectasis": "Respiratory",
    "Consolidation": "Respiratory Infection",
    "Infiltration": "Respiratory",
    "Emphysema": "COPD",
    "Fibrosis": "Interstitial Lung Disease",
    "Mass": "Oncology",
    "Nodule": "Oncology",
    "Pneumothorax": "Trauma/Emergency",
    "Pleural_Thickening": "Pleural Disease",
    "Hernia": "Hernia",
    "No Finding": "Normal",
}
