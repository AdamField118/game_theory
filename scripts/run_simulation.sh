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

# Description: Run game theory simulation with specified parameters
# Usage: sbatch scripts/run_simulation.sh
# Or with parameters: sbatch --export=DELTA=0.5,EPSILON=0 scripts/run_simulation.sh

echo "=========================================="
echo "GAME THEORY SIMULATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules and activate environment
module load miniconda3
source activate game_theory
echo "Environment activated"

# Set environment variables for optimal performance
export JAX_ENABLE_X64=True
# Removed problematic XLA_FLAGS that cause crashes

# Check GPU availability
echo "Checking GPU availability..."
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Set default parameters (can be overridden by SLURM --export)
DELTA=${DELTA:-0.5}
EPSILON=${EPSILON:-0.0}
VERTICES=${VERTICES:-300}
TIME_STEPS=${TIME_STEPS:-400}
TEMPERATURE=${TEMPERATURE:-100.0}
BETA=${BETA:-0.01}
SEED=${SEED:-42}

echo "Simulation Parameters:"
echo "  δ (delta): $DELTA"
echo "  ε (epsilon): $EPSILON"
echo "  Vertices: $VERTICES"
echo "  Time steps: $TIME_STEPS"
echo "  Temperature: $TEMPERATURE"
echo "  β (beta): $BETA"
echo "  Random seed: $SEED"
echo "=========================================="

# Create output directories
mkdir -p plots slurm_outfiles

echo "Starting simulation..."

# Run the simulation
python main.py \
    --delta $DELTA \
    --epsilon $EPSILON \
    --vertices $VERTICES \
    --time_steps $TIME_STEPS \
    --temperature $TEMPERATURE \
    --beta $BETA \
    --seed $SEED \
    --use_gpu \
    --save_data

RETURN_CODE=$?

echo "=========================================="
echo "Simulation completed with return code: $RETURN_CODE"
echo "End Time: $(date)"

# If successful, show output files
if [ $RETURN_CODE -eq 0 ]; then
    echo "Output files generated:"
    ls -la plots/*.png 2>/dev/null || echo "No plot files found"
    ls -la plots/data/ 2>/dev/null || echo "No data files found"
else
    echo "Simulation failed with return code: $RETURN_CODE"
    echo "Check error log: slurm_outfiles/evolutionary_game_${SLURM_JOB_ID}.err"
fi

echo "=========================================="
echo "JOB COMPLETE"
echo "=========================================="