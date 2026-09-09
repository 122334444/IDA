"""
kg_construction/chexzero_processor.py
──────────────────────────────────────
STEP 2 — Image Processing via CheXZero

CheXZero is a zero-shot CLIP-based model for chest X-ray pathology detection.
Paper: "Expert-level detection of pathologies from unannotated medical imaging
        data using self-supervised learning" (Tiu et al., Nature BME 2022)

What this module does:
  1. Load a chest X-ray image (.png / .dicom)
  2. Run CheXZero (CLIP) to get pathology probability scores for all 14 NIH labels
  3. Threshold the scores to decide positive findings
  4. Convert each positive finding into RDF triples for the KG

Usage:
    processor = CheXZeroProcessor()
    triples = processor.process_image("00000001_000.png", patient_id="P001")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import (
    CLIP_MODEL_NAME, CHEXZERO_THRESHOLD, NIH_LABELS, CHEXZERO_MODEL_PATH
)
from utils.triple_utils import (
    Triple, Predicates, patient_uri, disease_uri, finding_uri
)


# ── CheXZero text prompts ─────────────────────────────────────────────────────
# These are the zero-shot text templates used to describe each pathology.
# CheXZero computes cosine similarity between the image embedding and each
# text prompt embedding to score each pathology.

POSITIVE_TEMPLATES = [
    "Findings suggesting {label}.",
    "There is {label}.",
    "{label} is present.",
    "The presence of {label}.",
]

NEGATIVE_TEMPLATES = [
    "No {label}.",
    "No evidence of {label}.",
    "{label} is absent.",
    "There is no {label}.",
]


class CheXZeroProcessor:
    """
    Wraps CheXZero (CLIP-based) model to extract chest X-ray findings
    and convert them to RDF triples for the Knowledge Graph.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        threshold: float = CHEXZERO_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.threshold = threshold
        self.device = device or self._get_device()
        self.model = None
        self.preprocess = None
        self._load_model(model_name)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _get_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self, model_name: str):
        """Load CLIP model (used as CheXZero backbone)."""
        try:
            import clip
            import torch
            print(f"[CheXZero] Loading CLIP model '{model_name}' on {self.device} ...")
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()

            # If a fine-tuned CheXZero checkpoint exists, load it
            if CHEXZERO_MODEL_PATH.exists():
                print(f"[CheXZero] Loading fine-tuned weights from {CHEXZERO_MODEL_PATH}")
                state = torch.load(CHEXZERO_MODEL_PATH, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                print("[CheXZero] Fine-tuned weights loaded.")
            else:
                print("[CheXZero] No fine-tuned weights found — using base CLIP.")
                print(f"           (expected path: {CHEXZERO_MODEL_PATH})")
                print("           Download from: https://github.com/rajpurkarlab/CheXzero")

        except ImportError:
            print("[CheXZero] WARN: 'clip' package not installed.")
            print("           Install with: pip install git+https://github.com/openai/CLIP.git")
            print("           Falling back to mock scoring for testing.")
            self.model = None

    # ── Core inference ────────────────────────────────────────────────────────

    def score_image(self, image_path: Path) -> Dict[str, float]:
        """
        Score a chest X-ray for all 14 NIH pathologies.

        Returns:
            dict mapping label → probability (0–1)
        """
        if self.model is None:
            return self._mock_scores(image_path)

        import torch
        import clip

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        scores = {}
        with torch.no_grad():
            # Get image embedding
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            for label in NIH_LABELS:
                if label == "No Finding":
                    continue

                # Build positive and negative text prompts
                pos_texts = [t.format(label=label) for t in POSITIVE_TEMPLATES]
                neg_texts = [t.format(label=label) for t in NEGATIVE_TEMPLATES]
                all_texts = pos_texts + neg_texts

                # Tokenize and encode text
                tokens = clip.tokenize(all_texts, truncate=True).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Compute similarities
                sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                sims = sims[0].cpu().numpy()

                # Average positive template scores vs negative template scores
                n_pos = len(POSITIVE_TEMPLATES)
                pos_score = float(sims[:n_pos].mean())
                neg_score = float(sims[n_pos:].mean())

                # Normalize to 0–1
                scores[label] = pos_score / (pos_score + neg_score + 1e-8)

        return scores

    def _mock_scores(self, image_path: Path) -> Dict[str, float]:
        """
        Fallback mock scores when CLIP is not installed.
        Uses deterministic random seed based on filename for reproducibility.
        """
        import hashlib
        seed = int(hashlib.md5(image_path.name.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)

        scores = {}
        # Most images have "No Finding" or 1–2 findings
        for label in NIH_LABELS:
            if label == "No Finding":
                continue
            # Low probability for most labels (realistic distribution)
            scores[label] = float(rng.beta(0.5, 5.0))

        # If all scores are low, set "No Finding" scenario
        if max(scores.values()) < 0.3:
            scores = {k: v * 0.3 for k, v in scores.items()}

        return scores

    # ── Triple generation ─────────────────────────────────────────────────────

    def scores_to_triples(
        self,
        patient_id: str,
        scores: Dict[str, float],
        image_path: Optional[Path] = None,
    ) -> List[Triple]:
        """
        Convert pathology scores → RDF triples.

        Only positive findings (score ≥ threshold) become triples.
        Each finding generates 2–3 triples:
          (patient, hasCondition, disease)
          (disease, detectedBy, "CheXZero")
          (patient, hasFinding, finding)  [with confidence]
        """
        triples = []
        pat_uri = patient_uri(patient_id)

        for label, score in scores.items():
            if score < self.threshold:
                continue

            dis_uri = disease_uri(label)
            find_uri = finding_uri(label)

            # Core finding triple
            triples.append(Triple(
                subject=pat_uri,
                predicate=Predicates.HAS_CONDITION,
                obj=dis_uri,
                confidence=round(score, 3),
                source="chexzero",
                ontology="DO",
            ))

            # Detected by CheXZero
            triples.append(Triple(
                subject=dis_uri,
                predicate=Predicates.DETECTED_BY,
                obj="model:CheXZero",
                confidence=round(score, 3),
                source="chexzero",
            ))

            # If image path provided, link finding to image
            if image_path:
                triples.append(Triple(
                    subject=find_uri,
                    predicate=Predicates.VISIBLE_ON,
                    obj=f"image:{image_path.name}",
                    confidence=round(score, 3),
                    source="chexzero",
                ))

            # Severity approximation from score
            if score >= 0.8:
                severity = "severe"
            elif score >= 0.65:
                severity = "moderate"
            else:
                severity = "mild"

            triples.append(Triple(
                subject=dis_uri,
                predicate=Predicates.HAS_SEVERITY,
                obj=f"severity:{severity}",
                confidence=round(score, 3),
                source="chexzero",
            ))

        # If no positive findings
        if not triples:
            triples.append(Triple(
                subject=pat_uri,
                predicate=Predicates.HAS_FINDING,
                obj="finding:NoPathologyDetected",
                confidence=1.0,
                source="chexzero",
            ))

        return triples

    # ── Public entry point ────────────────────────────────────────────────────

    def process_image(
        self,
        image_path: Path,
        patient_id: str,
    ) -> Tuple[Dict[str, float], List[Triple]]:
        """
        Full pipeline: image file → (scores dict, triples list).

        Args:
            image_path : path to the chest X-ray .png or .dcm
            patient_id : unique patient identifier

        Returns:
            (scores, triples)
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"[CheXZero] Processing {image_path.name} for patient {patient_id} ...")
        scores = self.score_image(image_path)
        triples = self.scores_to_triples(patient_id, scores, image_path)

        # Print summary
        positives = {k: v for k, v in scores.items() if v >= self.threshold}
        if positives:
            print(f"[CheXZero] Positive findings ({len(positives)}):")
            for label, score in sorted(positives.items(), key=lambda x: -x[1]):
                print(f"           {label}: {score:.3f}")
        else:
            print(f"[CheXZero] No pathology detected above threshold {self.threshold}")

        print(f"[CheXZero] Generated {len(triples)} triples.")
        return scores, triples


# ── Convenience function ──────────────────────────────────────────────────────

def process_batch(
    image_paths: List[Path],
    patient_ids: List[str],
    threshold: float = CHEXZERO_THRESHOLD,
) -> Dict[str, List[Triple]]:
    """
    Process multiple images at once.

    Returns:
        dict mapping patient_id → list of triples
    """
    processor = CheXZeroProcessor(threshold=threshold)
    results = {}
    for img_path, pid in zip(image_paths, patient_ids):
        try:
            _, triples = processor.process_image(img_path, pid)
            results[pid] = triples
        except Exception as e:
            print(f"[CheXZero] ERROR processing {img_path}: {e}")
            results[pid] = []
    return results


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from config import DEMO_PATIENT, IMAGES_DIR

    print("=" * 60)
    print("CheXZero Processor — Test Run")
    print("=" * 60)

    # Look for any test image
    test_images = list(IMAGES_DIR.glob("*.png"))[:1] if IMAGES_DIR.exists() else []

    if test_images:
        img = test_images[0]
        processor = CheXZeroProcessor()
        scores, triples = processor.process_image(img, "TEST_001")
    else:
        print("[TEST] No real images found — creating a dummy test image ...")
        import tempfile
        dummy = Image.new("L", (224, 224), color=128)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            dummy.save(f.name)
            dummy_path = Path(f.name)

        processor = CheXZeroProcessor()
        scores, triples = processor.process_image(dummy_path, "TEST_001")
        dummy_path.unlink()

    print("\n── Scores ──")
    for label, score in sorted(scores.items(), key=lambda x: -x[1])[:5]:
        bar = "█" * int(score * 20)
        print(f"  {label:<25} {score:.3f}  {bar}")

    print("\n── Generated Triples ──")
    for t in triples:
        print(f"  {t}")
