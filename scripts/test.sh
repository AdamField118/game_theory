#!/bin/bash
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem-per-cpu=4g
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH -J test_evolutionary_game
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=your_email@wpi.edu
#SBATCH -o slurm_outfiles/test_%j.out
#SBATCH -e slurm_outfiles/test_%j.err

echo "=========================================="
echo "BASIC TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# Load environment
module load miniconda3
source activate game_theory
echo "Environment activated"

# Set basic JAX environment
export JAX_ENABLE_X64=True

# Check GPU availability
echo "Checking GPU..."
python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Create output directory
mkdir -p plots slurm_outfiles

echo "Running quick test simulation..."

# Run a SMALL test (50 vertices, 100 time steps)
python main.py \
    --delta 0.5 \
    --epsilon 0 \
    --vertices 50 \
    --time_steps 100 \
    --use_gpu \
    --save_data

RETURN_CODE=$?

echo "=========================================="
echo "Test completed with return code: $RETURN_CODE"
echo "End Time: $(date)"

if [ $RETURN_CODE -eq 0 ]; then
    echo "SUCCESS! Output files:"
    ls -la plots/*.png 2>/dev/null || echo "No plots found"
else
    echo "TEST FAILED - Check error log"
fi

echo "=========================================="