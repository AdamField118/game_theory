#!/bin/bash
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem-per-cpu=8g
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH -J evolutionary_game
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@wpi.edu
#SBATCH -o slurm_outfiles/evolutionary_game_%j.out
#SBATCH -e slurm_outfiles/evolutionary_game_%j.err

echo "=============================================="
echo "SEPARATED EVOLUTIONARY GAME SIMULATION"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start Time: $(date)"
echo "=============================================="

# Load modules and activate environment
module load miniconda3
source activate game_theory
echo "Environment activated"

# Set environment variables for optimal performance
export JAX_ENABLE_X64=True

# Check GPU availability
echo "Checking GPU availability..."
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# **GAME TYPE SELECTION** (NEW)
PAYOFF_SCHEME=${PAYOFF_SCHEME:-wealth_mediated}

# **SHARED PARAMETERS**
VERTICES=${VERTICES:-300}
TIME_STEPS=${TIME_STEPS:-400}
SEED=${SEED:-42}
PRESET=${PRESET:-}

# **WEALTH-MEDIATED SPECIFIC PARAMETERS**
DELTA=${DELTA:-0.5}
EPSILON=${EPSILON:-0.0}
TEMPERATURE=${TEMPERATURE:-100.0}
BETA=${BETA:-0.01}

# **ZERO-SUM SPECIFIC PARAMETERS**
INITIAL_PATTERN=${INITIAL_PATTERN:-random}
INITIAL_BANK_VALUE=${INITIAL_BANK_VALUE:-0.0}
STRIPE_SIZE=${STRIPE_SIZE:-}

echo "Simulation Parameters:"
echo "  Game type: $PAYOFF_SCHEME"
echo "  Vertices: $VERTICES"
echo "  Time steps: $TIME_STEPS"
echo "  Random seed: $SEED"

if [ "$PAYOFF_SCHEME" = "wealth_mediated" ]; then
    echo "  --- Wealth-Mediated Parameters ---"
    echo "  δ (tie bonus): $DELTA"
    echo "  ε (winning bonus): $EPSILON"
    echo "  Temperature: $TEMPERATURE"
    echo "  β (beta): $BETA"
elif [ "$PAYOFF_SCHEME" = "zero_sum" ]; then
    echo "  --- Zero-Sum Parameters ---"
    echo "  Initial pattern: $INITIAL_PATTERN"
    echo "  Initial bank value: $INITIAL_BANK_VALUE"
    if [ -n "$STRIPE_SIZE" ]; then
        echo "  Stripe size: $STRIPE_SIZE"
    fi
fi

if [ -n "$PRESET" ]; then
    echo "  Preset configuration: $PRESET"
fi
echo "=============================================="

# Create output directories
mkdir -p plots slurm_outfiles

echo "Starting simulation..."

# Build base command
CMD="python main.py \
    --payoff_scheme $PAYOFF_SCHEME \
    --vertices $VERTICES \
    --time_steps $TIME_STEPS \
    --seed $SEED \
    --use_gpu \
    --save_data"

# Add game-specific parameters
if [ "$PAYOFF_SCHEME" = "wealth_mediated" ]; then
    CMD="$CMD --delta $DELTA --epsilon $EPSILON --temperature $TEMPERATURE --beta $BETA"
elif [ "$PAYOFF_SCHEME" = "zero_sum" ]; then
    CMD="$CMD --initial_pattern $INITIAL_PATTERN --initial_bank_value $INITIAL_BANK_VALUE"
    if [ -n "$STRIPE_SIZE" ]; then
        CMD="$CMD --stripe_size $STRIPE_SIZE"
    fi
fi

# Add preset if specified (overrides individual parameters)
if [ -n "$PRESET" ]; then
    CMD="$CMD --preset $PRESET"
fi

# Run the simulation
echo "Executing: $CMD"
eval $CMD

RETURN_CODE=$?

echo "=============================================="
echo "Simulation completed with return code: $RETURN_CODE"
echo "End Time: $(date)"

# If successful, show output files
if [ $RETURN_CODE -eq 0 ]; then
    echo "Output files generated:"
    ls -la plots/*.png 2>/dev/null || echo "No plot files found"
    ls -la plots/data/ 2>/dev/null || echo "No data files found"
    ls -la plots/analysis/ 2>/dev/null || echo "No analysis files found"
else
    echo "Simulation failed with return code: $RETURN_CODE"
    echo "Check error log: slurm_outfiles/evolutionary_game_${SLURM_JOB_ID}.err"
fi

echo "=============================================="
echo "JOB COMPLETE"
echo "=============================================="