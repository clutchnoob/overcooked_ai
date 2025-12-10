#!/bin/bash
# Upload Run 4 results to Dropbox
# Usage: ./upload_to_dropbox.sh
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
echo "Uploading Run $RUN_NUM to Dropbox"
echo "=========================================="
echo "Dropbox destination: $DROPBOX_RUN_DIR"
echo ""

# Create Dropbox directory structure
echo "Creating Dropbox directories..."
dbxcli mkdir "$DROPBOX_RUN_DIR" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/logs" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/bc_runs" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/gail_runs" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_sp" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_bc" 2>/dev/null || true
dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_gail" 2>/dev/null || true
echo "Directories created."
echo ""

# Upload logs
echo "=== Uploading logs ==="
if [ -d "$SCRIPT_DIR/logs" ]; then
    for f in "$SCRIPT_DIR/logs"/*.out "$SCRIPT_DIR/logs"/*.err; do
        [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/logs/$(basename "$f")" 2>/dev/null && echo "  Uploaded: $(basename "$f")"
    done
else
    echo "  No logs directory found"
fi
echo ""

# Upload README and Results
echo "=== Uploading documentation ==="
[ -f "$SCRIPT_DIR/README.md" ] && dbxcli put "$SCRIPT_DIR/README.md" "$DROPBOX_RUN_DIR/README.md" && echo "  Uploaded: README.md"
[ -f "$SCRIPT_DIR/Run_${RUN_NUM}_Results.md" ] && dbxcli put "$SCRIPT_DIR/Run_${RUN_NUM}_Results.md" "$DROPBOX_RUN_DIR/Run_${RUN_NUM}_Results.md" && echo "  Uploaded: Run_${RUN_NUM}_Results.md"
echo ""

# Upload BC models
echo "=== Uploading BC models ==="
BC_DIR="$SRC_DIR/human_aware_rl/bc_runs_run${RUN_NUM}"
if [ -d "$BC_DIR" ]; then
    for split in train test; do
        if [ -d "$BC_DIR/$split" ]; then
            for layout_dir in "$BC_DIR/$split"/*/; do
                layout=$(basename "$layout_dir")
                dbxcli mkdir "$DROPBOX_RUN_DIR/models/bc_runs/$split/$layout" 2>/dev/null || true
                for f in "$layout_dir"/*; do
                    [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/bc_runs/$split/$layout/$(basename "$f")" 2>/dev/null
                done
                echo "  Uploaded: bc_runs/$split/$layout"
            done
        fi
    done
else
    echo "  BC directory not found: $BC_DIR"
fi
echo ""

# Upload GAIL models
echo "=== Uploading GAIL models ==="
GAIL_DIR="$SRC_DIR/human_aware_rl/gail_runs_run${RUN_NUM}"
if [ -d "$GAIL_DIR" ]; then
    for layout_dir in "$GAIL_DIR"/*/; do
        layout=$(basename "$layout_dir")
        dbxcli mkdir "$DROPBOX_RUN_DIR/models/gail_runs/$layout" 2>/dev/null || true
        for f in "$layout_dir"/*; do
            [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/gail_runs/$layout/$(basename "$f")" 2>/dev/null
        done
        echo "  Uploaded: gail_runs/$layout"
    done
else
    echo "  GAIL directory not found: $GAIL_DIR"
fi
echo ""

# Upload PPO_SP checkpoints (final checkpoint + config)
echo "=== Uploading PPO_SP checkpoints ==="
PPO_SP_DIR="$SRC_DIR/results/ppo_sp_run${RUN_NUM}"
if [ -d "$PPO_SP_DIR" ]; then
    for run_dir in "$PPO_SP_DIR"/*/; do
        run_name=$(basename "$run_dir")
        latest_ckpt=$(ls -d "$run_dir"/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$latest_ckpt" ]; then
            ckpt_name=$(basename "$latest_ckpt")
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_sp/$run_name" 2>/dev/null || true
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_sp/$run_name/$ckpt_name" 2>/dev/null || true
            [ -f "$run_dir/config.json" ] && dbxcli put "$run_dir/config.json" "$DROPBOX_RUN_DIR/models/ppo_sp/$run_name/config.json" 2>/dev/null
            for f in "$latest_ckpt"/*; do
                [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/ppo_sp/$run_name/$ckpt_name/$(basename "$f")" 2>/dev/null
            done
            echo "  Uploaded: ppo_sp/$run_name/$ckpt_name"
        fi
    done
else
    echo "  PPO_SP directory not found: $PPO_SP_DIR"
fi
echo ""

# Upload PPO_BC checkpoints (final checkpoint + config)
echo "=== Uploading PPO_BC checkpoints ==="
PPO_BC_DIR="$SRC_DIR/results/ppo_bc_run${RUN_NUM}"
if [ -d "$PPO_BC_DIR" ]; then
    for run_dir in "$PPO_BC_DIR"/*/; do
        run_name=$(basename "$run_dir")
        latest_ckpt=$(ls -d "$run_dir"/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$latest_ckpt" ]; then
            ckpt_name=$(basename "$latest_ckpt")
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_bc/$run_name" 2>/dev/null || true
            dbxcli mkdir "$DROPBOX_RUN_DIR/models/ppo_bc/$run_name/$ckpt_name" 2>/dev/null || true
            [ -f "$run_dir/config.json" ] && dbxcli put "$run_dir/config.json" "$DROPBOX_RUN_DIR/models/ppo_bc/$run_name/config.json" 2>/dev/null
            for f in "$latest_ckpt"/*; do
                [ -f "$f" ] && dbxcli put "$f" "$DROPBOX_RUN_DIR/models/ppo_bc/$run_name/$ckpt_name/$(basename "$f")" 2>/dev/null
            done
            echo "  Uploaded: ppo_bc/$run_name/$ckpt_name"
        fi
    done
else
    echo "  PPO_BC directory not found: $PPO_BC_DIR"
fi
echo ""

# Upload PPO_GAIL checkpoints (final checkpoint + config)
echo "=== Uploading PPO_GAIL checkpoints ==="
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
    echo "  PPO_GAIL directory not found: $PPO_GAIL_DIR"
fi

# Also check for ppo_gail_runs_run4 in human_aware_rl (alternative location)
PPO_GAIL_ALT_DIR="$SRC_DIR/human_aware_rl/ppo_gail_runs_run${RUN_NUM}"
if [ -d "$PPO_GAIL_ALT_DIR" ]; then
    echo "  Also checking alternative location: $PPO_GAIL_ALT_DIR"
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
fi
echo ""

echo "=========================================="
echo "Upload complete!"
echo "View at: $DROPBOX_RUN_DIR"
echo "=========================================="
