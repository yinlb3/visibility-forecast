#!/usr/bin/env bash
# Operational pipeline wrapper: run unified pipeline.py.
# Usage: bash /home/yjzx/code/vis/run_pipeline.sh [--utc|--bjt] [YYYYMMDDHH [YYYYMMDDHH]]
# Founded in 2026-07-28
# Modified in 2026-08-06
# @author: yinlb

# 0. Absolute path to the meteva conda env Python interpreter.
# Replace this with your actual path if meteva is installed elsewhere.
PYTHON="/root/miniconda3/envs/meteva/bin/python"

# 1. Switch to the script directory so relative config paths resolve
cd "$(dirname "$0")"

# 2. Run unified pipeline and propagate its exit code
echo "[run_pipeline] pipeline.py started"
"${PYTHON}" pipeline.py "$@"
pipe_status=$?
if [ ${pipe_status} -ne 0 ]; then
    echo "[run_pipeline] pipeline.py failed, exit code ${pipe_status}"
    exit ${pipe_status}
fi

echo "[run_pipeline] pipeline finished successfully"
