# Game Theory Simulation

A JAX-accelerated implementation of wealth-mediated thermodynamic strategy evolution for reproducing Figure 1 from:

**"Community Formation in Wealth-Mediated Thermodynamic Strategy Evolution"**  
*Connor Olson, Andrew Belmonte, Christopher Griffin (2022)*

## Setup Environment

```bash
conda env create -f environment.yml
```

## Usage

To run everything this repo has to offer, including:
- The two evolutions feature in figure 1
- Comparison graphs
- Analysis graphs including shannon entropy
You should submit this job:
```bash
sbatch scripts/run_figure1_batch.sh
```

In order to run a game with your chosen parameters submit:
```bash
sbatch --export=DELTA=0.5,EPSILON=0 scripts/run_simulation.sh
```

If you do not have SLURM, try:

```bash
# Stationary communities (δ=0.5, ε=0)
python main.py --delta 0.5 --epsilon 0

# Drifting communities (δ=0, ε=16)  
python main.py --delta 0 --epsilon 16

# Custom parameters
python main.py --delta 0.3 --epsilon 0.8 --vertices 500 --time_steps 600
```

## Output Files

### Plots
- `evolution_d{δ}_e{ε}.png`: Main evolution plot (Figure 1 style)
- `analysis_d{δ}_e{ε}.png`: Community analysis plots
- `comparison_plot.png`: Multi-parameter comparison
- `figure_1_reproduction.png`: Side-by-side stationary/drifting

### Data Files (if --save_data used)
- `simulation_d{δ}_e{ε}_{timestamp}.npz`: Raw simulation data
- `parameters_d{δ}_e{ε}_{timestamp}.json`: Parameters and metadata
- `summary_d{δ}_e{ε}.txt`: Summary statistics

## Results 

Here is the recreation of figure 1 by this code:
![comparison plot](./plots/analysis/comparison_plots.png)
Note: There is a [figure_1_reproduction.png](./plots/figure_1_reproduction.png) that is better formatted with a color bar and a title, but it currently is not formatted well.

Here are analysis plots of different evolutions:  
Analysis of the δ=0, ε=16 evolution:
![analysis of drifting](./plots/analysis/analysis_d0.0_e16.0.png)
Analysis of the δ=0.5, ε=0 evolution:
![analysis of stationary](./plots/analysis/analysis_d0.5_e0.0.png)

## References

Olson, C., Belmonte, A., & Griffin, C. (2022). Community Formation in Wealth-Mediated Thermodynamic Strategy Evolution. *arXiv preprint arXiv:2206.13160*.