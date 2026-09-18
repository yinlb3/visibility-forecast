@echo off
rem Operational pipeline wrapper: run unified pipeline.py.
rem Usage: run_pipeline.bat [--utc|--bjt] [YYYYMMDDHH [YYYYMMDDHH]]
rem Founded in 2026-07-28
rem Modified in 2026-08-06
rem @author: yinlb
setlocal

rem 0. Use the configured Python interpreter (absolute path to conda env)
set "PYTHON=C:\ProgramData\miniconda3\envs\meteva\python.exe"

rem 1. Switch to the script directory so relative config paths resolve
cd /d "%~dp0"

rem 2. Run unified pipeline and propagate its exit code
echo [run_pipeline] pipeline.py started
"%PYTHON%" pipeline.py %*
set pipe_status=%errorlevel%
if not "%pipe_status%"=="0" (
    echo [run_pipeline] pipeline.py failed, exit code %pipe_status%
    exit /b %pipe_status%
)

echo [run_pipeline] pipeline finished successfully
