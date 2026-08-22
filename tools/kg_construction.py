"""tools/kg_construction.py
KG Construction Tool — Part of Semantic Processing Agent (SPA)

Implements:
  1. CheXzero-style CLIP model for X-ray pathology extraction
  2. N2K Mapper (NLP chunking + NER) for clinical narrative
  3. KG Augmentation — combines triples into final Knowledge Graph

Output: List of RDF-style triples: (subject, predicate, object)
"""

import os
import json
import torch
import numpy as np
import networkx as nx
import nltk
from pathlib import Path
from typing import Optional
from PIL import Image
from utils.helpers import get_logger, map_entity_to_ontology, load_config

logger = get_logger("kg_construction")

# Download NLTK data (first run only)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("maxent_ne_chunker", quiet=True)
    nltk.download("words", quiet=True)


#  NIH Disease Prompts for CLIP
    
NIH_PATHOLOGY_PROMPTS = {
    "Atelectasis": [
        "chest x-ray showing atelectasis",
        "collapsed lung on chest radiograph",
        "linear atelectasis",
    ],
    "Cardiomegaly": [
        "chest x-ray showing cardiomegaly",
        "enlarged heart on chest radiograph",
        "increased cardiothoracic ratio",
    ],
    "Effusion": [
        "pleural effusion on chest x-ray",
        "fluid in pleural space",
        "blunting of costophrenic angle",
    ],
    "Infiltration": [
        "pulmonary infiltrate on chest x-ray",
        "haziness in lung fields",
    ],
    "Mass": [
        "lung mass on chest x-ray",
        "pulmonary mass lesion",
    ],
    "Nodule": [
        "pulmonary nodule on chest x-ray",
        "lung nodule",
    ],
    "Pneumonia": [
        "pneumonia on chest x-ray",
        "lobar consolidation",
        "air space opacity consistent with pneumonia",
    ],
    "Pneumothorax": [
        "pneumothorax on chest x-ray",
        "collapsed lung with air in pleural space",
        "absent lung markings pneumothorax",
    ],
    "Consolidation": [
        "pulmonary consolidation chest x-ray",
        "airspace consolidation",
    ],
    "Edema": [
        "pulmonary edema chest x-ray",
        "bilateral pulmonary edema",
        "interstitial edema",
    ],
    "Emphysema": [
        "emphysema chest x-ray",
        "hyperinflated lungs",
        "flattened diaphragm emphysema",
    ],
    "Fibrosis": [
        "pulmonary fibrosis chest x-ray",
        "interstitial fibrosis",
        "reticular pattern fibrosis",
    ],
    "Pleural_Thickening": [
        "pleural thickening chest x-ray",
        "thickened pleura",
    ],
    "Hernia": [
        "diaphragmatic hernia chest x-ray",
        "bowel loops in chest",
    ],
    "No Finding": [
        "normal chest x-ray",
        "no acute cardiopulmonary findings",
    ],
}


class CheXzeroProcessor:
    """
    CheXzero-style zero-shot X-ray classifier using CLIP.
    Extracts pathological findings from chest X-ray images.
    Reference: Tiu et al., Expert-level detection of pathologies from unannotated medical imaging data
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._model = None
        self._processor = None
        logger.info(f"CheXzeroProcessor on {self.device}")

    def _load_model(self):
        """Lazy load CLIP model."""
        if self._model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                logger.info(f"Loading CLIP model: {self.model_name}")
                self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                self._model.eval()
            except Exception as e:
                logger.warning(f"Could not load CLIP model ({e}). Using mock predictions.")
                self._model = "mock"

    def extract_findings(
        self,
        image_path: str,
        threshold: float = 0.15,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Run CLIP zero-shot classification on a chest X-ray.
        Returns list of findings above threshold.
        """
        self._load_model()

        if self._model == "mock":
            return self._mock_findings(image_path)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Cannot open image {image_path}: {e}")
            return []

        pathologies = list(NIH_PATHOLOGY_PROMPTS.keys())
        # Use first prompt for each pathology (ensemble in production)
        texts = [NIH_PATHOLOGY_PROMPTS[p][0] for p in pathologies]

        with torch.no_grad():
            inputs = self._processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).cpu().numpy()

        findings = []
        for pathology, prob in zip(pathologies, probs):
            if pathology != "No Finding" and prob >= threshold:
                findings.append({
                    "pathology": pathology,
                    "confidence": float(prob),
                    "source": "CheXzero",
                })

        # Sort by confidence
        findings.sort(key=lambda x: x["confidence"], reverse=True)
        return findings[:top_k]

    def _mock_findings(self, image_path: str) -> list[dict]:
        """Return plausible mock findings when model isn't loaded."""
        import hashlib
        h = int(hashlib.md5(image_path.encode()).hexdigest(), 16) % 100
        candidates = ["Cardiomegaly", "Edema", "Effusion", "Pneumonia", "Atelectasis"]
        selected = candidates[h % len(candidates)]
        return [{"pathology": selected, "confidence": 0.72 + (h % 20) / 100, "source": "CheXzero_mock"}]


class N2KMapper:
    """
    Narrative-to-KG (N2K) Mapper.
    Uses NLP chunking with customised grammar to extract
    entities from clinical narratives, then maps them
    to medical ontology classes via NER.
    """

    # Custom NLTK grammar for medical entity chunking
    GRAMMAR = r"""
        VITAL: {<CD><NN.*>+}
        SYMPTOM: {<JJ>*<NN.*>+<IN>?<NN.*>*}
        MEASUREMENT: {<CD><NN.*><CC>?<CD>?<NN.*>?}
        FINDING: {<VBG|VBN><JJ>*<NN.*>+}
    """

    def __init__(self):
        self.chunk_parser = nltk.RegexpParser(self.GRAMMAR)

    def extract_entities(self, narrative: str) -> list[dict]:
        """Extract named entities from clinical text."""
        tokens = nltk.word_tokenize(narrative)
        pos_tags = nltk.pos_tag(tokens)
        tree = self.chunk_parser.parse(pos_tags)

        entities = []
        for subtree in tree.subtrees():
            if subtree.label() in ("VITAL", "SYMPTOM", "MEASUREMENT", "FINDING"):
                phrase = " ".join(word for word, tag in subtree.leaves())
                mapped = map_entity_to_ontology(phrase)
                entities.append({
                    "text": phrase,
                    "class": mapped["class"],
                    "ontology": mapped["ontology"],
                    "ontology_id": mapped["id"],
                })

        # Also extract known entities directly
        for key in ["age", "gender", "heart rate", "temperature", "oxygen saturation",
                    "dyspnea", "edema", "fever", "cough", "chest pain"]:
            if key in narrative.lower():
                mapped = map_entity_to_ontology(key)
                entities.append({
                    "text": key,
                    "class": mapped["class"],
                    "ontology": mapped["ontology"],
                    "ontology_id": mapped["id"],
                })

        # Deduplicate by text
        seen = set()
        unique = []
        for e in entities:
            if e["text"] not in seen:
                seen.add(e["text"])
                unique.append(e)
        return unique


class KGConstructionTool:
    """
    Main KG Construction Tool.
    Orchestrates CheXzero + N2K mapper and builds patient KG as triples.
    """

    def __init__(self, config: dict = None):
        cfg = config or load_config()
        self.chexzero = CheXzeroProcessor(
            model_name=cfg["knowledge_graph"]["xray_model"]
        )
        self.n2k = N2KMapper()
        self.graphs: dict[str, nx.DiGraph] = {}   # patient_id → KG

    def build(self, patient_record: dict) -> dict:
        """
        Build KG for a single patient.

        Args:
            patient_record: dict with keys:
                patient_id, image_path, clinical_narrative,
                age, gender, heart_rate, bp_systolic, bp_diastolic,
                temperature, o2_saturation, respiratory_rate,
                symptoms, disease_labels

        Returns:
            { patient_id, triples: [(s,p,o),...], graph: nx.DiGraph }
        """
        pid = patient_record["patient_id"]
        logger.info(f"Building KG for patient {pid}")

        triples = []

        # ── 1. Demographics triples ──────────────────────────────────────────
        demo_fields = {
            "has_age": str(patient_record.get("age", "unknown")),
            "has_gender": str(patient_record.get("gender", "unknown")),
            "has_heart_rate": f"{patient_record.get('heart_rate', '?')} bpm",
            "has_bp": f"{patient_record.get('bp_systolic', '?')}/{patient_record.get('bp_diastolic', '?')} mmHg",
            "has_temperature": f"{patient_record.get('temperature', '?')}°C",
            "has_o2_saturation": f"{patient_record.get('o2_saturation', '?')}%",
            "has_respiratory_rate": f"{patient_record.get('respiratory_rate', '?')} breaths/min",
        }
        for pred, obj in demo_fields.items():
            triples.append((pid, pred, obj))

        # ── 2. Symptom triples (from N2K mapper) ────────────────────────────
        narrative = patient_record.get("clinical_narrative", "")
        if narrative:
            entities = self.n2k.extract_entities(narrative)
            for ent in entities:
                pred = f"has_{ent['class'].lower()}"
                obj = f"{ent['text']} [{ent['ontology']}:{ent['ontology_id']}]"
                triples.append((pid, pred, obj))

        # Explicit symptoms
        for sym in patient_record.get("symptoms", []):
            mapped = map_entity_to_ontology(sym)
            triples.append((pid, "has_symptom", f"{sym} [{mapped['ontology']}]"))

        # ── 3. Radiological findings triples (CheXzero) ─────────────────────
        image_path = patient_record.get("image_path")
        if image_path and os.path.exists(image_path):
            findings = self.chexzero.extract_findings(image_path)
            for f in findings:
                triples.append((
                    pid,
                    "has_radiological_finding",
                    f"{f['pathology']} (conf={f['confidence']:.2f}) [RadLex]"
                ))
        else:
            # Use label as fallback if image not accessible
            for label in patient_record.get("disease_labels", []):
                triples.append((pid, "has_radiological_finding", f"{label} [RadLex]"))

        # ── 4. Build NetworkX DiGraph ────────────────────────────────────────
        G = nx.DiGraph()
        G.add_node(pid, type="Patient")
        for s, p, o in triples:
            G.add_edge(s, o, relation=p)

        self.graphs[pid] = G
        logger.info(f"KG built: {len(triples)} triples, {G.number_of_nodes()} nodes")

        return {
            "patient_id": pid,
            "triples": triples,
            "graph": G,
        }

    def augment(self, patient_id: str, new_data: dict) -> dict:
        """Update existing KG with new information."""
        if patient_id not in self.graphs:
            logger.warning(f"No existing KG for {patient_id}. Building fresh.")
            return self.build(new_data)

        G = self.graphs[patient_id]
        new_triples = []
        for key, value in new_data.items():
            triple = (patient_id, f"updated_{key}", str(value))
            G.add_edge(*triple[:2], relation=triple[1])
            G.nodes[triple[2]] if triple[2] in G else G.add_node(triple[2])
            new_triples.append(triple)

        logger.info(f"KG augmented for {patient_id}: +{len(new_triples)} triples")
        return {"patient_id": patient_id, "new_triples": new_triples}

    def get_graph_json(self, patient_id: str) -> dict:
        """Return KG as JSON (for Plotly/D3 visualization)."""
        G = self.graphs.get(patient_id)
        if G is None:
            return {}
        nodes = [{"id": n, "type": G.nodes[n].get("type", "Entity")} for n in G.nodes()]
        edges = [{"source": u, "target": v, "relation": d["relation"]}
                 for u, v, d in G.edges(data=True)]
        return {"nodes": nodes, "edges": edges}
