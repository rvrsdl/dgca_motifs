# Creating Network Motifs with Developmental Graph Cellular Automata (ALife 2024)
This repository contains the code used in the following paper presented at the Artificial Life 2024 conference:

R. Waldegrave, S. Stepney, and M. A. Trefzer, ‘Creating Network Motifs with Developmental Graph Cellular Automata’, presented at the ALIFE 2024: Proceedings of the 2024 Artificial Life Conference, MIT Press, Jul. 2024. doi: 10.1162/isal_a_00734.


## Contents
- `dgcav2_demo.m`: A minimal MATLAB/Octave implementation of the DGCA v2 system, using the Kroneckor tensor poduct graph update procedure as described in the paper. Just designed to illustrate how the system works
- `src`
    - `src/dgca.py`: A more complete Python implementation of the DGCA v2 system, which was used to run the experiments in the paper.
    - `src/evolve.py`: An implementation of the Microbial Genetic Algorithm for the DGCA SLP weights, including the use of separate "chromosomes" as described in the paper.
    - `src/motifs.py`: Various useful functions for obtaining lists of motifs etc.
- `motif_random_search.py` - Runs the random search experiment.
- `motif_evol_search.py` - Runs the evolutionary search experiment.
- `plot_biologicial_tsp.py` - Used in creating Figure 3 in the paper.
- `plot_random_search_results.py` - Used in creating Figure 7 in the paper.
- `plot_evol_search_results.py` - Used in creating Figures 8 & 9 in the paper.


## Installing requirements
- If using conda, create a new conda environment based on the environment.yml file:
    `conda env create --name motifs --file environment.yml`
- To use the environment do:
    `conda activate motifs`
- This project uses the Python library graph-tool which cannot be installed using `pip` because it has C++ dependencies. See here for details: https://graph-tool.skewed.de/installation.html
    - For this reason it is much easier to install all the dependecies using conda, as shown above.

## Examples (basic DGCA usage)
1. Running a few steps of DGCA and looking at the graphs at each step. (Easiest to do this in interactive mode or an ipython notebook)
```
from src.dgca import DGCA, GraphDef

# Set up your "seed graph" by provideing an adjacancy matrix (n*n) and state matrix (n*s)
g0 = GraphDef(np.array([[1,1,0],[0,0,1],[1,0,0]]), np.array([[1,0],[0,1],[0,1]]))
# Initialise the DGCA model with default settings. 
# SLP weights will be assigned randomly (re-initialise to get new random weights.)
dgca = DGCA(num_states=g0.num_states)

# View the seed_graph (nodes are coloured by state)
g0.draw_gt()

# Apply an update step
g1 = dgca.step(g0)

# View the updated graph 
g1.draw_graph()

# (If all nodes have been removed the graph will not be displayed)
```

2. Running the system for lots of steps in one go. The runner object will stop the run early if the system has entered an attractor (ie. a graph has been seen before in the run), or if the graph gets too big.
```
import numpy as np
import matplotlib.pyplot as plt
from src.dgca import DGCA, GraphDef, Runner

seed_graph = GraphDef(np.array([[1,1,0],[0,0,1],[1,0,0]]), np.array([[1,0],[0,1],[0,1]]))
dgca = DGCA(num_states=seed_graph.num_states)
runner = Runner(max_steps=200, max_size=128)

final_graph = runner.run(dgca, seed_graph)

# Plot the size (number of nodes) in the graph at each timestep over the course of the run.
plt.plot(runner.graph_size())
```

For running the motif experiments in the paper, see the scripts `motif_random_search.py` and `motif_evol_search.py`.
