@echo off
rem Operational pipeline wrapper: run preprocess.py then inference.py.
rem Usage: run_pipeline.bat [YYYYMMDDHH [YYYYMMDDHH]]
rem Founded in 2026-07-28
rem Modified in 2026-07-31
rem @author: yinlb
setlocal

rem 0. Use the configured Python interpreter (absolute path to conda env)
set "PYTHON=C:\ProgramData\miniconda3\envs\meteva\python.exe"

rem 1. Switch to the script directory so relative config paths resolve
cd /d "%~dp0"

rem 2. Run preprocessing; stop the pipeline on failure
echo [run_pipeline] preprocess.py started
"%PYTHON%" preprocess.py %*
set pre_status=%errorlevel%
if not "%pre_status%"=="0" (
    echo [run_pipeline] preprocess.py failed, exit code %pre_status%
    exit /b %pre_status%
)

rem 3. Run inference and propagate its exit code
echo [run_pipeline] inference.py started
"%PYTHON%" inference.py %*
set inf_status=%errorlevel%
if not "%inf_status%"=="0" (
    echo [run_pipeline] inference.py failed, exit code %inf_status%
    exit /b %inf_status%
)

echo [run_pipeline] pipeline finished successfully
