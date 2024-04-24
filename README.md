# Creating Network Motifs with Developmental Graph Cellular Automata (ALife 2024)
This repository will contain the code used in a paper submitted to the ALife 2024 conference.

## Contents
- `dgcav2_demo.m`: A minimal MATLAB/Octave implementation of the DGCA v2 system, using the Kroneckor tensor poduct graph update procedure as described in the paper. Just designed to illustrate how the system works
- `dgca.py`: A more complete Python implementation of the DGCA v2 system, which was used to run the experiments in the paper.
- `evolve.py`: An implementation of the Microbial Genetic Algorithm for the DGCA SLP weights, including the use of separate "chromosomes" as described in the paper.
- `motif_random_search.py` - Runs the random search experiment.
- `motif_evol_search.py` - Runs the evolutionary search experiment.
- `plot_biologicial_tsp.py` - Used in creating Figure 3 in the paper.
- `plot_random_search_results.py` - Used in creating Figure 7 in the paper.
- `plot_evol_search_results.py` - Used in creating Figures 8 & 9 in the paper.

**N.B. this is currently incomplete.**


