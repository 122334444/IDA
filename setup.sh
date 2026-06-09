#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Intelligent Doctor Assistant — Setup Script
#  Run: bash setup.sh
# ═══════════════════════════════════════════════════════════════════
set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       Intelligent Doctor Assistant — Setup               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Load .env ────────────────────────────────────────────────────
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    echo "✅ .env loaded"
else
    echo "❌ No .env file found."
    exit 1
fi

# ── Python check ─────────────────────────────────────────────────
PYTHON=$(which python3 || which python)
PY_VERSION=$($PYTHON --version 2>&1)
echo "✅ Python: $PY_VERSION"

# ── Virtual environment ──────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate (works on Mac/Linux/Windows Git Bash)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi
echo "✅ Virtual environment activated: $(which python)"

# ── Upgrade pip ──────────────────────────────────────────────────
pip install --upgrade pip setuptools wheel -q
echo "✅ pip upgraded"

# ── Install core packages first (no version pins) ────────────────
echo ""
echo "📦 Installing core dependencies..."

# Flask + web
pip install flask flask-cors python-dotenv pyyaml requests tqdm -q
echo "  ✅ Flask ready"

# Data science
pip install "numpy>=1.26" "pandas>=2.2" scikit-learn matplotlib plotly Pillow -q
echo "  ✅ Data science libs ready"

# NLP
pip install nltk networkx -q
echo "  ✅ NLP libs ready"

# Kaggle
pip install kaggle -q
echo "  ✅ Kaggle CLI ready"

# HuggingFace ecosystem (heavy — install last)
echo "  📥 Installing HuggingFace stack (this takes ~2 min)..."
pip install "transformers>=4.42" "accelerate>=0.31" "peft>=0.11" \
            "datasets>=2.20" "trl>=0.9" huggingface_hub sentencepiece -q
echo "  ✅ HuggingFace stack ready"

# PyTorch — auto-detect best version for this Python/OS
echo "  📥 Installing PyTorch (auto-selecting best version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null || \
pip install torch torchvision -q
echo "  ✅ PyTorch ready"

# bitsandbytes (optional — only needed for GPU fine-tuning)
pip install bitsandbytes -q 2>/dev/null && echo "  ✅ bitsandbytes ready" || \
echo "  ⚠️  bitsandbytes skipped (GPU fine-tuning only — OK for local dev)"

# pypdf, chromadb
pip install pypdf chromadb -q 2>/dev/null || echo "  ⚠️  chromadb skipped"

echo ""
echo "✅ All packages installed"

# ── NLTK data ────────────────────────────────────────────────────
echo ""
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
pkgs = ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','words']
for p in pkgs:
    nltk.download(p, quiet=True)
print('✅ NLTK data ready')
"

# ── Kaggle credentials ───────────────────────────────────────────
echo ""
echo "🔑 Setting up Kaggle credentials..."
mkdir -p ~/.kaggle
if [ -n "$KAGGLE_API_TOKEN" ]; then
    echo "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
    chmod 600 ~/.kaggle/access_token
    echo "✅ Kaggle token saved"
else
    echo "⚠️  KAGGLE_API_TOKEN not set in .env"
fi

# ── HuggingFace login ────────────────────────────────────────────
echo ""
echo "🤗 HuggingFace login..."
if [ -n "$HF_TOKEN" ]; then
    python3 -c "
from huggingface_hub import login
import os
try:
    login(token='$HF_TOKEN', add_to_git_credential=False)
    print('✅ HuggingFace logged in')
except Exception as e:
    print(f'⚠️  HF login: {e}')
" 2>/dev/null || echo "⚠️  HF login skipped"
fi

# ── Download NIH Chest X-Ray dataset ────────────────────────────
echo ""
echo "⬇️  Downloading NIH Chest X-Ray dataset (~40GB, this will take a while)..."
mkdir -p data/raw/nih_xray

if [ -f "data/raw/nih_xray/Data_Entry_2017.csv" ]; then
    echo "✅ NIH data already downloaded"
else
    python3 -c "
import os, subprocess
token = os.environ.get('KAGGLE_API_TOKEN','')
if not token:
    print('⚠️  No Kaggle token — skipping download')
    exit(0)
os.environ['KAGGLE_API_TOKEN'] = token
result = subprocess.run(
    ['kaggle','datasets','download','-d','nih-chest-xrays/data','-p','data/raw/','--unzip'],
    capture_output=True, text=True
)
if result.returncode == 0:
    print('✅ NIH dataset downloaded and extracted')
else:
    print('⚠️  Download failed:', result.stderr[:300])
    print('   → Download manually: https://www.kaggle.com/datasets/nih-chest-xrays/data')
    print('   → Extract to: data/raw/nih_xray/')
" 2>&1
fi

# ── Preprocess ───────────────────────────────────────────────────
echo ""
if [ -f "data/raw/nih_xray/Data_Entry_2017.csv" ]; then
    echo "⚙️  Preprocessing dataset..."
    python3 utils/preprocess.py && echo "✅ Data preprocessed"
else
    echo "⚠️  NIH data not found — run after downloading:"
    echo "    python utils/preprocess.py"
fi

# ── Create dirs ──────────────────────────────────────────────────
mkdir -p models/llama3_finetuned data/vector_db data/medical_guidelines data/processed

# ── Run tests ────────────────────────────────────────────────────
echo ""
echo "🧪 Running pipeline tests..."
python3 tests/test_pipeline.py 2>/dev/null | grep -E "TEST|passed|✅|❌" || true

# ── Done ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup complete!                                      ║"
echo "║                                                          ║"
echo "║  Start the app:  python app.py                           ║"
echo "║  Open browser:   http://localhost:5001                   ║"
echo "║                                                          ║"
echo "║  For LLaMA-3 fine-tuning → run Kaggle notebook          ║"
echo "║  (Upload .ipynb to kaggle.com, GPU T4, Run All)         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
