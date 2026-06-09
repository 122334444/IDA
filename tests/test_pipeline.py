"""tests/test_pipeline.py — Full pipeline integration test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ida_orchestrator import IDAOrchestrator
from utils.helpers import load_config

def run_tests():
    print("\n" + "═"*60)
    print("  IDA Pipeline Tests")
    print("═"*60)

    cfg = load_config()
    ida = IDAOrchestrator(cfg)
    passed = 0

    # ── Test 1: CHF patient ──────────────────────────────────────
    print("\n[TEST 1] CHF patient — full explain pipeline")
    chf = {
        "patient_id": "TEST_CHF", "age": 72, "gender": "M",
        "heart_rate": 108, "bp_systolic": 145, "bp_diastolic": 92,
        "temperature": 37.1, "o2_saturation": 89, "respiratory_rate": 22,
        "symptoms": ["dyspnea on exertion", "bilateral leg edema", "orthopnea"],
        "clinical_narrative": "72M with worsening dyspnea and bilateral edema.",
        "disease_labels": ["Cardiomegaly", "Edema", "Effusion"], "image_path": ""
    }
    r = ida.run(chf, action="explain")
    assert r["status"] == "success", f"FAIL: {r}"
    assert r["diagnosis"]["primary"] == "Congestive Heart Failure", f"FAIL: got {r['diagnosis']['primary']}"
    assert r["knowledge_graph"]["triple_count"] > 10
    assert len(r["explanation"]) > 200
    print(f"  ✅ Diagnosis: {r['diagnosis']['primary']} ({r['diagnosis']['confidence']:.0%})")
    print(f"  ✅ KG triples: {r['knowledge_graph']['triple_count']}")
    print(f"  ✅ Explanation: {len(r['explanation'])} chars")
    passed += 1

    # ── Test 2: Pneumonia patient ────────────────────────────────
    print("\n[TEST 2] Pneumonia patient — predict only")
    pne = {
        "patient_id": "TEST_PNE", "age": 45, "gender": "F",
        "heart_rate": 98, "bp_systolic": 118, "bp_diastolic": 76,
        "temperature": 38.9, "o2_saturation": 93, "respiratory_rate": 20,
        "symptoms": ["productive cough", "fever", "pleuritic chest pain"],
        "clinical_narrative": "45F with 4-day fever and productive cough.",
        "disease_labels": ["Pneumonia", "Consolidation"], "image_path": ""
    }
    r2 = ida.run(pne, action="predict")
    assert r2["status"] == "success"
    assert r2["diagnosis"]["primary"] == "Pneumonia", f"FAIL: got {r2['diagnosis']['primary']}"
    print(f"  ✅ Diagnosis: {r2['diagnosis']['primary']} ({r2['diagnosis']['confidence']:.0%})")
    passed += 1

    # ── Test 3: KG only ──────────────────────────────────────────
    print("\n[TEST 3] KG-only mode")
    r3 = ida.run(chf, action="kg_only")
    assert r3["status"] == "success"
    assert "diagnosis" not in r3
    assert r3["knowledge_graph"]["triple_count"] > 0
    print(f"  ✅ KG triples: {r3['knowledge_graph']['triple_count']}")
    passed += 1

    # ── Test 4: Normal patient ───────────────────────────────────
    print("\n[TEST 4] Normal patient")
    norm = {
        "patient_id": "TEST_NORM", "age": 34, "gender": "M",
        "heart_rate": 72, "bp_systolic": 118, "bp_diastolic": 76,
        "temperature": 36.7, "o2_saturation": 98, "respiratory_rate": 14,
        "symptoms": ["routine checkup"],
        "clinical_narrative": "34M routine checkup, no complaints.",
        "disease_labels": ["No Finding"], "image_path": ""
    }
    r4 = ida.run(norm, action="predict")
    assert r4["status"] == "success"
    print(f"  ✅ Diagnosis: {r4['diagnosis']['primary']}")
    passed += 1

    print("\n" + "═"*60)
    print(f"  {passed}/4 tests passed ✅")
    print("═"*60 + "\n")

if __name__ == "__main__":
    run_tests()
