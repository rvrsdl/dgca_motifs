
#%%
import os
from functools import partial
from datetime import datetime
import numpy as np
import zstandard # Not used directly but need to import before graph_tool to prevent a bug.
import graph_tool.all as gt
from src.dgca import DGCA, GraphDef, Runner
from src.evolve import EvolvableDGCA, SignificanceProfileFitness, ChromosomalMGA
from src.motifs import generate_named_triads

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
#%%
# Various settings
POPULATION_SIZE = 30
MUTATE_RATE = 0.02
CROSS_RATE = 0.5
CROSS_STYLE = 'cols'
NUM_TRIALS = 5000
NUM_SHUFFLES = 500
RAND_ESU = [1.0,1.0,0.5]
TSP = {
    'ecoli': np.array([-0.613, -0.377, -0.313, 0.620, -0.010, -0.015, -0.001, -0.001, -0.000, 0.000, 0.000, 0.000, 0.000])
}
timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
results_fn = os.path.join(PROJ_DIR,'results', f'results_{timestr}.csv')
conditions = {'max_size': 300, 
              'min_size': 100, 
              'min_connectivity': 0.002, 
              'max_connectivity': 0.04, 
              'min_component_frac': 0.4}
motifs13 = list(generate_named_triads().values())
zscore_func = partial(gt.motif_significance, k=3, n_shuffles=NUM_SHUFFLES, p=RAND_ESU, motif_list=motifs13)
fitness_fn = SignificanceProfileFitness(target_sig_prof=TSP['ecoli'], 
                                        zscore_func=zscore_func,
                                        conditions=conditions,
                                        verbose=True)
seed_graph = GraphDef(A=np.array([[0]]), S=np.array([[1,0,0]]), num_states=3)
model = EvolvableDGCA(num_states=seed_graph.num_states)
runner = Runner(max_steps=100, max_size=300)
mga = ChromosomalMGA(popsize=POPULATION_SIZE,
                     seed_graph=seed_graph,
                     model=model,
                     runner=runner,
                     fitness_fn=fitness_fn,
                     mutate_rate=MUTATE_RATE,
                     cross_rate=CROSS_RATE,
                     cross_style=CROSS_STYLE,
                     csv_filename=results_fn)
#%%
mga.run(steps=NUM_TRIALS)

# %%
