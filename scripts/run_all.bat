@echo off
REM Run all notebooks headlessly on Windows

echo ==========================================
echo Running CausalBench notebooks headlessly
echo ==========================================

REM Create results directory
if not exist results mkdir results
if not exist output mkdir output

REM Log file
set LOG_FILE=results\logs.txt
echo %date% %time%: Starting notebook execution > "%LOG_FILE%"

REM List of notebooks
set NOTEBOOKS=notebook_ihdp notebook_twins notebook_sachs notebook_acic notebook_lalonde

REM Run notebooks using Python script (handles paths better)
echo Running notebooks...
python scripts\run_notebooks.py >> "%LOG_FILE%" 2>&1
if %errorlevel% equ 0 (
    echo Notebooks execution completed
) else (
    echo Some notebooks may have failed - check logs
)

REM Generate summary PDF
echo Generating summary PDF...
python scripts\generate_summary.py

REM Create zip archive using Python (more reliable)
echo Creating archive...
python -c "import zipfile, os; from pathlib import Path; zf = zipfile.ZipFile('output/causalbench.zip', 'w'); exclude = {'.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints', 'output', '.venv', 'env'}; [zf.write(f, f) for r, d, files in os.walk('.') for f in [os.path.join(r, file) for file in files] if not any(x in f for x in exclude) and not f.endswith(('.pyc', '.zip', '.png')) and not (f.startswith('data') and f.endswith('.csv')) and not (f.startswith('results') and f.endswith(('.png', '.csv'))) and not f.startswith('output')]; zf.close(); print('Archive created')" 2>>"%LOG_FILE%"

echo.
echo ==========================================
echo Summary
echo ==========================================
echo Results saved to: results\
echo Archive created: output\causalbench.zip

REM Print SHA256 if available
where certutil >nul 2>&1
if %errorlevel% equ 0 (
    certutil -hashfile output\causalbench.zip SHA256 | findstr /V "SHA256" | findstr /V "CertUtil"
)

echo %date% %time%: All notebooks processed >> "%LOG_FILE%"
