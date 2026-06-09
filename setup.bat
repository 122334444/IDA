@echo off
REM ═══════════════════════════════════════════════════════
REM  Intelligent Doctor Assistant — Windows Setup
REM  Run: setup.bat
REM ═══════════════════════════════════════════════════════

echo.
echo ╔══════════════════════════════════════════════╗
echo ║   Intelligent Doctor Assistant — Setup      ║
echo ╚══════════════════════════════════════════════╝
echo.

REM Load .env
if not exist .env (
    echo ERROR: .env file not found. Create it from .env.example
    pause
    exit /b 1
)

REM Read HF_TOKEN and KAGGLE token from .env
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if "%%a"=="HF_TOKEN" set HF_TOKEN=%%b
    if "%%a"=="KAGGLE_API_TOKEN" set KAGGLE_API_TOKEN=%%b
)

REM Virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo [OK] Dependencies installed

REM NLTK data
python -c "import nltk; [nltk.download(p,quiet=True) for p in ['punkt','punkt_tab','averaged_perceptron_tagger_eng','words']]; print('[OK] NLTK ready')"

REM Kaggle credentials
if not exist %USERPROFILE%\.kaggle mkdir %USERPROFILE%\.kaggle
echo %KAGGLE_API_TOKEN%> %USERPROFILE%\.kaggle\access_token
echo [OK] Kaggle token saved

REM HuggingFace login
python -c "from huggingface_hub import login; import os; login(token='%HF_TOKEN%', add_to_git_credential=False); print('[OK] HuggingFace logged in')" 2>nul || echo [WARN] HF login failed

REM Download NIH data
echo Downloading NIH Chest X-Ray dataset...
if not exist data\raw\nih_xray mkdir data\raw\nih_xray
set KAGGLE_API_TOKEN=%KAGGLE_API_TOKEN%
kaggle datasets download -d nih-chest-xrays/data -p data\raw\ --unzip 2>nul || echo [WARN] Download failed - download manually from kaggle.com/datasets/nih-chest-xrays/data

REM Create dirs
if not exist models\llama3_finetuned mkdir models\llama3_finetuned
if not exist data\vector_db mkdir data\vector_db
if not exist data\medical_guidelines mkdir data\medical_guidelines

REM Preprocess
if exist data\raw\nih_xray\Data_Entry_2017.csv (
    echo Preprocessing dataset...
    python utils\preprocess.py
    echo [OK] Data preprocessed
) else (
    echo [WARN] NIH data not found. Run: python utils\preprocess.py after downloading.
)

echo.
echo ╔══════════════════════════════════════════════╗
echo ║  Setup complete!                             ║
echo ║  Start: python app.py                        ║
echo ║  Open:  http://localhost:5000                ║
echo ╚══════════════════════════════════════════════╝
echo.
pause
