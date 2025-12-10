#!/bin/bash
# Replace old PPO_GAIL files in Dropbox with new rerun results
# Usage: ./replace_ppo_gail_dropbox.sh
#
# Prerequisites:
#   - dbxcli must be installed and authenticated
#   - Activate the dropbox_cli conda environment first:
#     source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
#     conda activate /om2/user/mabdel03/conda_envs/dropbox_cli

set -e

RUN_NUM=4
DROPBOX_BASE="/All files/Mahmoud Abdelmoneum/6.S890/Test_Runs"
DROPBOX_RUN_DIR="$DROPBOX_BASE/Run_$RUN_NUM"

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
SRC_DIR="$PROJECT_ROOT/src"

echo "=========================================="
echo "Replacing PPO_GAIL files in Dropbox"
echo "=========================================="
echo "Dropbox destination: $DROPBOX_RUN_DIR"
echo ""

# Step 1: Delete old PPO_GAIL models
echo "=== Step 1: Deleting old PPO_GAIL models ==="
echo "Deleting $DROPBOX_RUN_DIR/models/ppo_gail..."
dbxcli rm -r "$DROPBOX_RUN_DIR/models/ppo_gail" 2>/dev/null || echo "  (directory may not exist)"
echo "Done."
echo ""

# Step 2: Delete old PPO_GAIL logs
echo "=== Step 2: Deleting old PPO_GAIL logs ==="
# List and delete old ppo_gail logs (without run4 subdirectory)
for f in $(dbxcli ls "$DROPBOX_RUN_DIR/logs" 2>/dev/null | grep -E "^ppo_gail_" || true); do
    echo "  Deleting: $f"
    dbxcli rm "$DROPBOX_RUN_DIR/logs/$f" 2>/dev/null || true
done
echo "Done."
echo ""

# Step 3: Recreate directory structure
echo "=== Step 3: Creating new directory structure ==="
dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_gail" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/logs/ppo_gail_run4" 2>/dev/null || true
echo "Done."
echo ""

# Step 4: Upload new PPO_GAIL checkpoints
echo "=== Step 4: Uploading new PPO_GAIL checkpoints ==="
PPO_GAIL_DIR="$SRC_DIR/results/ppo_gail_run${RUN_NUM}"
if [ -d "$PPO_GAIL_DIR" ]; then
    for run_dir in "$PPO_GAIL_DIR"/*/; do
        run_name=$(basename "$run_dir")
        latest_ckpt=$(ls -d "$run_dir"/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$latest_ckpt" ]; then
            ckpt_name=$(basename "$latest_ckpt")
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_gail/$run_name" 2>/dev/null || true
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_gail/$run_name/$ckpt_name" 2>/dev/null || true
            [ -f "$run_dir/config.json" ] && dbxcli put "$run_dir/config.json" "$DROPBOX_RUN_DIR/models/ppo_gail/$run_name/config.json" 2>/dev/null
            for f in "$latest_ckpt"/*; do
                [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/ppo_gail/$run_name/$ckpt_name/$(basename "$f")" 2>/dev/null
            done
            echo "  Uploaded: ppo_gail/$run_name/$ckpt_name"
        fi
    done
else
    echo "  PPO_GAIL checkpoint directory not found: $PPO_GAIL_DIR"
fi
echo ""

# Step 5: Upload new PPO_GAIL final models
echo "=== Step 5: Uploading new PPO_GAIL final models ==="
PPO_GAIL_ALT_DIR="$SRC_DIR/human_aware_rl/ppo_gail_runs_run${RUN_NUM}"
if [ -d "$PPO_GAIL_ALT_DIR" ]; then
    for layout_dir in "$PPO_GAIL_ALT_DIR"/*/; do
        layout=$(basename "$layout_dir")
        for seed_dir in "$layout_dir"/*/; do
            seed=$(basename "$seed_dir")
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_gail/$layout/$seed" 2>/dev/null || true
            for f in "$seed_dir"/*; do
                [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/ppo_gail/$layout/$seed/$(basename "$f")" 2>/dev/null
            done
            echo "  Uploaded: ppo_gail/$layout/$seed"
        done
    done
else
    echo "  PPO_GAIL final models directory not found: $PPO_GAIL_ALT_DIR"
fi
echo ""

# Step 6: Upload new PPO_GAIL logs
echo "=== Step 6: Uploading new PPO_GAIL logs ==="
LOG_DIR="$SCRIPT_DIR/logs/ppo_gail_run4"
if [ -d "$LOG_DIR" ]; then
    for f in "$LOG_DIR"/*.out "$LOG_DIR"/*.err; do
        if [ -f "$f" ]; then
            dbxcli put "$f" "$DROPBOX_RUN_DIR/logs/ppo_gail_run4/$(basename "$f")" 2>/dev/null
            echo "  Uploaded: $(basename "$f")"
        fi
    done
else
    echo "  Log directory not found: $LOG_DIR"
fi
echo ""

echo "=========================================="
echo "PPO_GAIL replacement complete!"
echo "View at: $DROPBOX_RUN_DIR"
echo "=========================================="
