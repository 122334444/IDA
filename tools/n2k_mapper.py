"""
kg_construction/n2k_mapper.py
──────────────────────────────
STEP 3 — N2K Mapper: Clinical Narrative → Knowledge Graph Triples

N2K = "Narrative to Knowledge"

This module implements the N2K mapper from the paper, which uses:
  1. Tokenization + POS tagging
  2. Chunking with customised grammar (noun-phrase grammar rules)
  3. Phrase extraction for clinically relevant entities
  4. Mapping extracted phrases to triple subjects/objects

These extracted phrases are then passed to NER (Step 4) for ontology assignment.

Usage:
    mapper = N2KMapper()
    phrases = mapper.extract_phrases(text)
    triples = mapper.text_to_triples(text, patient_id="P001")
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import MAX_TEXT_LENGTH
from utils.triple_utils import (
    Triple, Predicates,
    patient_uri, disease_uri, symptom_uri, anatomy_uri, vital_uri
)


# ── Clinical Grammar Rules ────────────────────────────────────────────────────
# These regex-based chunk rules identify medically relevant noun phrases
# from tokenised clinical text.
# Grammar adapted for clinical NLP — captures things like:
#   "bilateral pleural effusion", "right lower lobe consolidation",
#   "elevated troponin", "shortness of breath"

CLINICAL_CHUNK_GRAMMAR = r"""
    # Multi-word anatomical + finding phrases
    ANAT_FINDING: {<JJ>*<NN.*>+<IN>?<JJ>*<NN.*>+}

    # Simple noun phrases
    NP: {<DT>?<JJ>*<NN.*>+}

    # Adjective + noun (e.g. "bilateral", "right-sided")
    QUAL_NP: {<JJ><NN.*>}

    # Verb + noun (e.g. "reveals opacity", "shows consolidation")
    VP_NP: {<VBZ|VBD|VBP><DT>?<JJ>*<NN.*>+}
"""

# ── Clinical keyword patterns ─────────────────────────────────────────────────
# Direct regex patterns for vital signs and demographics embedded in text.

VITALS_PATTERNS = {
    "temperature": [
        r"(?:temperature|temp)[^\d]*(\d+\.?\d*)\s*[°℃CF]",
        r"T[:\s]+(\d+\.?\d*)[°\s℃CF]",
        r"(\d+\.?\d*)\s*°[CF]",
    ],
    "respiratory_rate": [
        r"(?:respiratory rate|RR|resp\.? rate)[^\d]*(\d+)",
        r"(\d+)\s*breaths?\s*(?:per|/)\s*min",
    ],
    "heart_rate": [
        r"(?:heart rate|HR|pulse)[^\d]*(\d+)",
        r"(\d+)\s*bpm",
        r"(\d+)\s*beats?\s*(?:per|/)\s*min",
    ],
    "blood_pressure": [
        r"(?:BP|blood pressure)[:\s]*(\d+)\s*/\s*(\d+)",
    ],
    "spo2": [
        r"(?:SpO2|O2 sat|oxygen sat)[^\d]*(\d+)\s*%?",
        r"(\d+)\s*%\s*(?:on|at)\s*room air",
        r"saturation[^\d]*(\d+)",
    ],
    "troponin": [
        r"[Tt]roponin[^\d]*(\d+\.?\d*)\s*(?:ng|μg|ug)?\/?(?:mL|ml|dL)?",
    ],
}

# Clinical keywords categorized by semantic type
DISEASE_KEYWORDS = {
    "pneumonia", "atelectasis", "effusion", "edema", "emphysema",
    "fibrosis", "cardiomegaly", "pneumothorax", "consolidation",
    "infiltration", "pleural thickening", "nodule", "mass", "hernia",
    "tuberculosis", "copd", "asthma", "bronchitis", "pleurisy",
    "pericarditis", "heart failure", "pulmonary embolism",
}

SYMPTOM_KEYWORDS = {
    "dyspnea", "shortness of breath", "cough", "fever", "fatigue",
    "chest pain", "hemoptysis", "wheezing", "sputum", "tachycardia",
    "tachypnea", "hypoxia", "pleuritic pain", "night sweats",
    "weight loss", "malaise", "anorexia", "cyanosis", "clubbing",
}

ANATOMY_KEYWORDS = {
    "lung", "chest", "heart", "pleura", "diaphragm", "bronchus",
    "trachea", "rib", "lobe", "hilum", "mediastinum", "pericardium",
    "parenchyma", "airway", "alveoli", "thorax", "carina",
    "right lower lobe", "left lower lobe", "right upper lobe",
    "left upper lobe", "bilateral",
}

MODIFIER_KEYWORDS = {
    "bilateral", "unilateral", "right", "left", "upper", "lower",
    "mild", "moderate", "severe", "acute", "chronic", "progressive",
    "new", "resolving", "worsening", "stable",
}


class N2KMapper:
    """
    Maps clinical narrative text to knowledge graph triples using
    NLP chunking with customised clinical grammar.
    """

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self.nlp = None
        self._load_nlp()

    # ── NLP model loading ─────────────────────────────────────────────────────

    def _load_nlp(self):
        """Load spaCy biomedical model or fall back to NLTK."""
        if self.use_spacy:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_sci_md")
                    print("[N2K] Loaded spaCy model: en_core_sci_md")
                except OSError:
                    # Try generic English model
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                        print("[N2K] Loaded spaCy model: en_core_web_sm (generic)")
                        print("[N2K] TIP: For better biomedical NLP, run:")
                        print("           pip install scispacy")
                        print("           pip install https://s3-us-west-2.amazonaws.com/"
                              "ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz")
                    except OSError:
                        print("[N2K] No spaCy model found. Falling back to NLTK.")
                        self.nlp = None
            except ImportError:
                print("[N2K] spaCy not installed. Falling back to NLTK.")
                self.nlp = None

        if self.nlp is None:
            self._init_nltk()

    def _init_nltk(self):
        """Ensure NLTK data is available."""
        try:
            import nltk
            for resource in ["punkt", "averaged_perceptron_tagger", "maxent_ne_chunker", "words"]:
                try:
                    nltk.data.find(f"tokenizers/{resource}")
                except LookupError:
                    nltk.download(resource, quiet=True)
        except ImportError:
            pass

    # ── Phrase extraction ─────────────────────────────────────────────────────

    def extract_phrases(self, text: str) -> Dict[str, List[str]]:
        """
        Extract clinically relevant phrases from text.

        Returns:
            dict with keys: "diseases", "symptoms", "anatomy", "modifiers", "vitals"
        """
        text = text[:MAX_TEXT_LENGTH].lower()

        result = {
            "diseases": [],
            "symptoms": [],
            "anatomy": [],
            "modifiers": [],
            "vitals": {},
        }

        # ── Keyword matching ──────────────────────────────────────────────────
        # Simple but robust: scan for known clinical keywords

        for kw in DISEASE_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                result["diseases"].append(kw)

        for kw in SYMPTOM_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                result["symptoms"].append(kw)

        for kw in ANATOMY_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                result["anatomy"].append(kw)

        for kw in MODIFIER_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                result["modifiers"].append(kw)

        # ── Vital sign extraction ─────────────────────────────────────────────
        result["vitals"] = self._extract_vitals(text)

        # ── spaCy noun chunks (if available) ──────────────────────────────────
        if self.nlp is not None:
            extra = self._spacy_extract(text)
            # Add spaCy-found entities that aren't already in keyword lists
            for entity in extra.get("diseases", []):
                if entity not in result["diseases"]:
                    result["diseases"].append(entity)
            for entity in extra.get("symptoms", []):
                if entity not in result["symptoms"]:
                    result["symptoms"].append(entity)

        return result

    def _spacy_extract(self, text: str) -> Dict[str, List[str]]:
        """Use spaCy NER + noun chunks to find additional entities."""
        doc = self.nlp(text[:10000])  # spaCy has length limits
        result = {"diseases": [], "symptoms": []}

        for ent in doc.ents:
            label = ent.label_.upper()
            ent_text = ent.text.lower().strip()
            if label in ("DISEASE", "CONDITION", "DISORDER"):
                result["diseases"].append(ent_text)
            elif label in ("SYMPTOM", "SIGN"):
                result["symptoms"].append(ent_text)

        return result

    def _extract_vitals(self, text: str) -> Dict[str, float]:
        """
        Extract numerical vital sign values from clinical text.

        e.g. "Temperature 37.8°C, Respiratory Rate 22 breaths/min"
        → {"temperature": 37.8, "respiratory_rate": 22}
        """
        vitals = {}
        for vital_name, patterns in VITALS_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        vitals[vital_name] = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
        return vitals

    # ── Demographic extraction ────────────────────────────────────────────────

    def extract_demographics(self, text: str) -> Dict[str, str]:
        """Extract age, gender from clinical narrative."""
        demo = {}

        # Age
        age_match = re.search(
            r'(\d+)[\s-]*(?:year|yr)s?[\s-]*old|age[d\s:]+(\d+)', text, re.IGNORECASE
        )
        if age_match:
            demo["age"] = age_match.group(1) or age_match.group(2)

        # Gender
        if re.search(r'\b(male|man|boy|M)\b', text, re.IGNORECASE):
            demo["gender"] = "male"
        elif re.search(r'\b(female|woman|girl|F)\b', text, re.IGNORECASE):
            demo["gender"] = "female"

        return demo

    # ── Triple generation ─────────────────────────────────────────────────────

    def text_to_triples(
        self,
        text: str,
        patient_id: str,
        vitals_dict: Optional[Dict] = None,
    ) -> List[Triple]:
        """
        Convert clinical narrative text → RDF triples.

        Args:
            text       : clinical notes / narrative
            patient_id : unique patient identifier
            vitals_dict: optional pre-parsed vitals dict (overrides text extraction)

        Returns:
            List of Triple objects
        """
        triples: List[Triple] = []
        pat_uri = patient_uri(patient_id)

        # Extract phrases
        phrases = self.extract_phrases(text)
        demo = self.extract_demographics(text)
        vitals = vitals_dict or phrases.get("vitals", {})

        # ── Demographic triples ───────────────────────────────────────────────
        if demo.get("age"):
            triples.append(Triple(
                subject=pat_uri,
                predicate=Predicates.HAS_AGE,
                obj=f"value:{demo['age']}",
                source="n2k",
            ))
        if demo.get("gender"):
            triples.append(Triple(
                subject=pat_uri,
                predicate=Predicates.HAS_GENDER,
                obj=f"demographic:{demo['gender']}",
                source="n2k",
            ))

        # ── Disease triples ───────────────────────────────────────────────────
        for disease in phrases["diseases"]:
            triples.append(Triple(
                subject=pat_uri,
                predicate=Predicates.HAS_CONDITION,
                obj=disease_uri(disease),
                confidence=0.85,
                source="n2k",
                ontology="DO",
            ))

        # ── Symptom triples ───────────────────────────────────────────────────
        for symptom in phrases["symptoms"]:
            triples.append(Triple(
                subject=pat_uri,
                predicate=Predicates.HAS_SYMPTOM,
                obj=symptom_uri(symptom),
                confidence=0.9,
                source="n2k",
                ontology="SYMP",
            ))

        # ── Anatomy triples ───────────────────────────────────────────────────
        for anat in phrases["anatomy"]:
            triples.append(Triple(
                subject=pat_uri,
                predicate=Predicates.HAS_ANATOMY,
                obj=anatomy_uri(anat),
                confidence=0.9,
                source="n2k",
                ontology="FMA",
            ))

        # ── Disease × Anatomy relations ───────────────────────────────────────
        # e.g. "right lower lobe consolidation" →
        #   (disease:consolidation, locatedIn, anatomy:right_lower_lobe)
        for disease in phrases["diseases"]:
            for anat in phrases["anatomy"]:
                triples.append(Triple(
                    subject=disease_uri(disease),
                    predicate=Predicates.LOCATED_IN,
                    obj=anatomy_uri(anat),
                    confidence=0.6,
                    source="n2k",
                ))

        # ── Vital sign triples ────────────────────────────────────────────────
        predicate_map = {
            "temperature":      Predicates.HAS_TEMPERATURE,
            "respiratory_rate": Predicates.HAS_RESP_RATE,
            "spo2":             Predicates.HAS_SPO2,
            "troponin":         Predicates.HAS_TROPONIN,
            "heart_rate":       Predicates.HAS_VITAL,
            "blood_pressure":   Predicates.HAS_VITAL,
        }
        for vital_name, vital_val in vitals.items():
            predicate = predicate_map.get(vital_name, Predicates.HAS_VITAL)
            triples.append(Triple(
                subject=pat_uri,
                predicate=predicate,
                obj=f"vital:{vital_name}:{vital_val}",
                confidence=1.0,
                source="n2k",
            ))

        print(f"[N2K] Patient {patient_id}: {len(triples)} triples from clinical text.")
        print(f"      Diseases: {phrases['diseases']}")
        print(f"      Symptoms: {phrases['symptoms']}")
        print(f"      Vitals:   {vitals}")

        return triples


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import DEMO_PATIENT

    print("=" * 60)
    print("N2K Mapper — Test Run")
    print("=" * 60)

    mapper = N2KMapper()
    text = DEMO_PATIENT["clinical_text"]
    triples = mapper.text_to_triples(
        text,
        patient_id=DEMO_PATIENT["patient_id"],
        vitals_dict=DEMO_PATIENT["vitals"],
    )

    print("\n── Generated Triples ──")
    for t in triples:
        print(f"  {t}  [conf={t.confidence:.2f}, src={t.source}]")
