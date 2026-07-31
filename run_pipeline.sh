#!/usr/bin/env bash
# Operational pipeline wrapper: run preprocess.py then inference.py.
# Usage: bash /home/yjzx/code/vis/run_pipeline.sh [YYYYMMDDHH [YYYYMMDDHH]]
# Founded in 2026-07-28
# Modified in 2026-07-31
# @author: yinlb

# 0. Absolute path to the meteva conda env Python interpreter.
# Replace this with your actual path if meteva is installed elsewhere.
PYTHON="/root/miniconda3/envs/meteva/bin/python"

# 1. Switch to the script directory so relative config paths resolve
cd "$(dirname "$0")"

# 2. Run preprocessing; stop the pipeline on failure
echo "[run_pipeline] preprocess.py started"
"${PYTHON}" preprocess.py "$@"
pre_status=$?
if [ ${pre_status} -ne 0 ]; then
    echo "[run_pipeline] preprocess.py failed, exit code ${pre_status}"
    exit ${pre_status}
fi

# 3. Run inference and propagate its exit code
echo "[run_pipeline] inference.py started"
"${PYTHON}" inference.py "$@"
inf_status=$?
if [ ${inf_status} -ne 0 ]; then
    echo "[run_pipeline] inference.py failed, exit code ${inf_status}"
    exit ${inf_status}
fi

echo "[run_pipeline] pipeline finished successfully"
