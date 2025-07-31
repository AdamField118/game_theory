#!/bin/bash
#SBATCH -o slurm_outfiles/figure1_%j.out
#SBATCH -e slurm_outfiles/figure1_%j.err

# Batch script to reproduce both scenarios from Figure 1
# Left plot: δ=0.5, ε=0 (stationary communities)
# Right plot: δ=0, ε=16 (drifting communities)

echo "=========================================="
echo "REPRODUCING FIGURE 1 FROM OLSON ET AL."
echo "=========================================="

# Create necessary directories
mkdir -p slurm_outfiles plots

# Job 1: Stationary communities (δ=0.5, ε=0)
echo "Submitting Job 1: Stationary Communities (δ=0.5, ε=0)"
JOB1=$(sbatch --export=DELTA=0.5,EPSILON=0.0,VERTICES=300,TIME_STEPS=400 \
              --job-name=stationary_communities \
              scripts/run_simulation.sh | awk '{print $4}')
echo "Job ID: $JOB1"

# Job 2: Drifting communities (δ=0, ε=16)
echo "Submitting Job 2: Drifting Communities (δ=0, ε=16)"
JOB2=$(sbatch --export=DELTA=0.0,EPSILON=16.0,VERTICES=300,TIME_STEPS=400 \
              --job-name=drifting_communities \
              scripts/run_simulation.sh | awk '{print $4}')
echo "Job ID: $JOB2"

echo "=========================================="
echo "Jobs submitted successfully!"
echo ""
echo "Expected outputs:"
echo "  - plots/evolution_d0.5_e0.0.png (Stationary)"
echo "  - plots/evolution_d0.0_e16.0.png (Drifting)"
echo "  - plots/data/ (Simulation data files)"
echo "=========================================="

# Submit a dependent job to create comparison plot
echo "Submitting comparison job (dependent on completion of both simulations)..."
COMPARISON_JOB=$(sbatch --dependency=afterok:$JOB1:$JOB2 \
                        --job-name=comparison_plot \
                        --partition=short \
                        --time=00:30:00 \
                        --mem=4G \
                        --output=slurm_outfiles/comparison_plot_%j.out \
                        --error=slurm_outfiles/comparison_plot_%j.err \
                        --wrap="python scripts/create_comparison.py")
echo "Comparison job ID: $COMPARISON_JOB"

echo "All jobs submitted."

# Generate analysis plots such as stategy over time.
python scripts/generate_analysis.py
