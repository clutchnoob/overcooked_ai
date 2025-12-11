#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Submit array job and capture job ID
ARRAY_JOB=$(sbatch --parsable eval_run3_parallel.sh)
echo "Submitted array job: $ARRAY_JOB"

# Submit merge job with dependency (runs after ALL array tasks succeed)
MERGE_JOB=$(sbatch --parsable --dependency=afterok:$ARRAY_JOB merge_run3.sh)
echo "Submitted merge job: $MERGE_JOB (will run after $ARRAY_JOB completes)"
