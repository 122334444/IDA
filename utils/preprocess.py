"""utils/preprocess.py
Preprocesses NIH Chest X-Ray dataset for IDA system.

Steps:
  1. Load Data_Entry_2017.csv
  2. Parse multi-label disease annotations
  3. Split into train/val/test
  4. Generate patient clinical records (simulate demographics + vitals)
  5. Save processed splits as Parquet files
"""

import os
import ast
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.helpers import load_config, get_logger, NIH_DISEASES, DISEASE_CATEGORY

logger = get_logger("preprocess")


# ─── Simulate Patient Demographics & Vitals ───────────────────────────────────

def simulate_clinical_data(row: pd.Series, seed: int = 42) -> dict:
    """
    NIH dataset provides only image + disease labels.
    We simulate plausible demographic + vital sign data
    for the structured EHR portion of the KG.
    In production, replace with real EHR data (MIMIC-IV etc.)
    """
    rng = np.random.default_rng(seed + hash(row["Image Index"]) % 2**31)

    disease_labels = row["disease_list"]
    is_chf = any(d in disease_labels for d in ["Cardiomegaly", "Edema", "Effusion"])
    is_infection = any(d in disease_labels for d in ["Pneumonia", "Consolidation", "Infiltration"])

    # Age: heart disease skews older
    age = int(rng.normal(65 if is_chf else 50, 15))
    age = max(18, min(95, age))

    gender = rng.choice(["M", "F"])

    # Vitals: abnormal if CHF or infection
    hr = int(rng.normal(100 if is_chf else 80, 15))
    hr = max(40, min(180, hr))

    bp_sys = int(rng.normal(140 if is_chf else 120, 15))
    bp_sys = max(70, min(220, bp_sys))
    bp_dia = int(rng.normal(90 if is_chf else 80, 10))

    temp = round(rng.normal(38.5 if is_infection else 36.8, 0.5), 1)
    temp = max(35.0, min(42.0, temp))

    o2_sat = int(rng.normal(90 if is_chf else 97, 3))
    o2_sat = max(70, min(100, o2_sat))

    rr = int(rng.normal(24 if is_infection else 16, 4))
    rr = max(8, min(40, rr))

    # Chief complaints / symptoms
    symptom_pool_chf = ["dyspnea on exertion", "orthopnea", "paroxysmal nocturnal dyspnea",
                         "bilateral leg edema", "fatigue", "decreased exercise tolerance"]
    symptom_pool_inf = ["productive cough", "fever", "chills", "chest pain", "sputum production"]
    symptom_pool_gen = ["shortness of breath", "cough", "chest discomfort", "malaise"]

    symptoms = []
    if is_chf:
        symptoms += rng.choice(symptom_pool_chf, size=min(3, len(symptom_pool_chf)), replace=False).tolist()
    if is_infection:
        symptoms += rng.choice(symptom_pool_inf, size=min(2, len(symptom_pool_inf)), replace=False).tolist()
    if not symptoms:
        symptoms = rng.choice(symptom_pool_gen, size=2, replace=False).tolist()

    # Clinical narrative
    narrative = (
        f"Patient is a {age}-year-old {('male' if gender=='M' else 'female')} presenting with "
        f"{', '.join(symptoms[:-1]) + ' and ' + symptoms[-1] if len(symptoms) > 1 else symptoms[0]}. "
        f"Vital signs: HR {hr} bpm, BP {bp_sys}/{bp_dia} mmHg, "
        f"Temperature {temp}°C, O2 Saturation {o2_sat}%, RR {rr} breaths/min."
    )

    return {
        "patient_id": row["Image Index"].replace(".png", ""),
        "image_path": row["Image Index"],
        "age": age,
        "gender": gender,
        "heart_rate": hr,
        "bp_systolic": bp_sys,
        "bp_diastolic": bp_dia,
        "temperature": temp,
        "o2_saturation": o2_sat,
        "respiratory_rate": rr,
        "symptoms": symptoms,
        "clinical_narrative": narrative,
        "disease_labels": disease_labels,
        "primary_diagnosis": disease_labels[0] if disease_labels else "No Finding",
        "disease_category": DISEASE_CATEGORY.get(
            disease_labels[0] if disease_labels else "No Finding", "Unknown"
        ),
        "follow_up_number": int(row.get("Follow-up #", 0)),
    }


# ─── Main Preprocessing ───────────────────────────────────────────────────────

def preprocess(config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)
    ds_cfg = cfg["dataset"]

    labels_csv = ds_cfg["labels_csv"]
    out_dir = Path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading NIH labels from {labels_csv}")
    df = pd.read_csv(labels_csv)

    # Parse multi-label string "Cardiomegaly|Edema" → list
    df["disease_list"] = df["Finding Labels"].apply(
        lambda s: [x.strip() for x in s.split("|") if x.strip() != "No Finding"]
        if s.strip() != "No Finding" else []
    )

    # Filter to target diseases (keep patients with at least 1 target OR No Finding)
    target = set(ds_cfg["target_diseases"])
    df["has_target"] = df["disease_list"].apply(
        lambda dl: any(d in target for d in dl) or len(dl) == 0
    )
    df_filtered = df[df["has_target"]].copy().reset_index(drop=True)

    logger.info(f"Total records after filtering: {len(df_filtered)}")

    # Generate clinical records
    logger.info("Simulating patient clinical data ...")
    records = [simulate_clinical_data(row) for _, row in df_filtered.iterrows()]
    df_records = pd.DataFrame(records)

    # Train/val/test split (stratified by primary diagnosis)
    train_ratio = ds_cfg["train_ratio"]
    val_ratio = ds_cfg["val_ratio"]

    train_df, temp_df = train_test_split(
        df_records,
        test_size=(1 - train_ratio),
        random_state=42,
        stratify=df_records["primary_diagnosis"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["primary_diagnosis"]
    )

    logger.info(f"Split — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    # Save label distribution
    dist = df_records["primary_diagnosis"].value_counts().to_dict()
    with open(out_dir / "distribution.json", "w") as f:
        json.dump(dist, f, indent=2)

    logger.info(f"Preprocessing complete. Files saved to {out_dir}")
    logger.info(f"Disease distribution: {dist}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    preprocess(args.config)
