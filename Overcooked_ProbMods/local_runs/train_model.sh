#!/bin/bash
###############################################################################
# SINGLE MODEL TRAINING WRAPPER
###############################################################################
# Train a single probabilistic model on a single layout.
#
# Usage:
#   ./train_model.sh <model_type> <layout> [options]
#
# Examples:
#   ./train_model.sh bayesian_bc cramped_room
#   ./train_model.sh bayesian_ppo_gail asymmetric_advantages --epochs 100
#
# Models:
#   Imitation: bayesian_bc, rational_agent, hierarchical_bc
#   RL: bayesian_gail, bayesian_ppo_bc, bayesian_ppo_gail
#
# Layouts:
#   cramped_room, asymmetric_advantages, coordination_ring
###############################################################################

set -e

# Source configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

###############################################################################
# ARGUMENT PARSING
###############################################################################

MODEL="$1"
LAYOUT="$2"
shift 2 || true

# Parse optional arguments
EPOCHS="${IMITATION_EPOCHS}"
TIMESTEPS="${RL_TIMESTEPS}"
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$MODEL" ] || [ -z "$LAYOUT" ]; then
    echo "Usage: $0 <model_type> <layout> [--epochs N] [--timesteps N] [--verbose]"
    echo ""
    echo "Models: ${ALL_MODELS[*]}"
    echo "Layouts: ${LAYOUTS[*]}"
    exit 1
fi

###############################################################################
# ENVIRONMENT SETUP
###############################################################################

# Activate Python environment
activate_environment

# Change to project directory
cd "$PROJECT_DIR"

# Set up logging
TIMESTAMP=$(get_timestamp)
LOG_FILE=$(get_log_path "$MODEL" "$LAYOUT" "$TIMESTAMP")

###############################################################################
# TRAINING EXECUTION
###############################################################################

section "TRAINING: $MODEL on $LAYOUT"
echo "Device:     $DEVICE"
echo "Log file:   $LOG_FILE"
echo "Timestamp:  $TIMESTAMP"
echo ""

# Log system info
{
    echo "=============================================================="
    echo "TRAINING LOG: $MODEL on $LAYOUT"
    echo "=============================================================="
    echo "Timestamp:  $(date)"
    echo "Device:     $DEVICE"
    echo "Model:      $MODEL"
    echo "Layout:     $LAYOUT"
    echo "Epochs:     $EPOCHS (for imitation models)"
    echo "Timesteps:  $TIMESTEPS (for RL models)"
    echo "Python:     $(which python)"
    echo "PyTorch:    $(python -c 'import torch; print(torch.__version__)')"
    echo "=============================================================="
    echo ""
} > "$LOG_FILE"

# Run the appropriate training script
case $MODEL in
    "bayesian_bc")
        log "Training Bayesian Behavioral Cloning..."
        python -u scripts/train_bayesian_bc.py \
            --layout "$LAYOUT" \
            --epochs "$EPOCHS" \
            --results_dir ./results \
            2>&1 | tee -a "$LOG_FILE"
        ;;
    
    "rational_agent")
        log "Training Rational Agent Model..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.rational_agent import RationalAgentConfig, RationalAgentTrainer

config = RationalAgentConfig(
    layout_name='$LAYOUT',
    num_epochs=$EPOCHS,
    results_dir='./results'
)
trainer = RationalAgentTrainer(config)
trainer.train()
print('Training complete!')
" 2>&1 | tee -a "$LOG_FILE"
        ;;
    
    "hierarchical_bc")
        log "Training Hierarchical BC Model..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.hierarchical_bc import HierarchicalBCConfig, HierarchicalBCTrainer

config = HierarchicalBCConfig(
    layout_name='$LAYOUT',
    num_epochs=$EPOCHS,
    num_goals=4,
    results_dir='./results'
)
trainer = HierarchicalBCTrainer(config)
trainer.train()
print('Training complete!')
" 2>&1 | tee -a "$LOG_FILE"
        ;;
    
    "bayesian_gail")
        log "Training Bayesian GAIL..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.bayesian_gail import BayesianGAILConfig, BayesianGAILTrainer

config = BayesianGAILConfig(
    layout_name='$LAYOUT',
    total_timesteps=$TIMESTEPS,
    results_dir='./results'
)
trainer = BayesianGAILTrainer(config)
trainer.train()
print('Training complete!')
" 2>&1 | tee -a "$LOG_FILE"
        ;;
    
    "bayesian_ppo_bc")
        log "Training Bayesian PPO+BC..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.bayesian_ppo_bc import BayesianPPOBCConfig, BayesianPPOBCTrainer

config = BayesianPPOBCConfig(
    layout_name='$LAYOUT',
    total_timesteps=$TIMESTEPS,
    results_dir='./results'
)
trainer = BayesianPPOBCTrainer(config)
trainer.train()
print('Training complete!')
" 2>&1 | tee -a "$LOG_FILE"
        ;;
    
    "bayesian_ppo_gail")
        log "Training Bayesian PPO+GAIL..."
        python -u -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
from probmods.models.bayesian_ppo_gail import BayesianPPOGAILConfig, BayesianPPOGAILTrainer

config = BayesianPPOGAILConfig(
    layout_name='$LAYOUT',
    total_timesteps=$TIMESTEPS,
    results_dir='./results'
)
trainer = BayesianPPOGAILTrainer(config)
trainer.train()
print('Training complete!')
" 2>&1 | tee -a "$LOG_FILE"
        ;;
    
    *)
        error "Unknown model: $MODEL"
        echo "Available models: ${ALL_MODELS[*]}"
        exit 1
        ;;
esac

# Log completion
{
    echo ""
    echo "=============================================================="
    echo "TRAINING COMPLETE"
    echo "Finished at: $(date)"
    echo "=============================================================="
} >> "$LOG_FILE"

log "Training complete! Results saved to: $RESULTS_DIR/$MODEL/$LAYOUT/"
log "Log file: $LOG_FILE"

