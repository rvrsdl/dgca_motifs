"""
Runs the random search experiment
"""
import os
import numpy as np
import graph_tool.all as gt
from tqdm import tqdm
import pickle
from datetime import datetime
from src.dgca import DGCA, GraphDef, Runner
from src.evolve import check_conditions
from src.motifs import generate_named_triads

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

VERBOSE = False
NTRIALS = 100
N_SHUFFLES = 500
RAND_ESU = [0.9, 0.8, 0.7]
seed_graph = GraphDef(A=np.array([[0]]), S=np.array([[1,0,0]]), num_states=3)

runner = Runner(max_steps=256, max_size=300)
conditions = {'max_size': 300, 
              'min_size': 100, 
              'min_connectivity': 0.002, 
              'max_connectivity': 0.03, 
              'min_component_frac': 0.4}

motifs13 = list(generate_named_triads().values())
results = []
pbar = tqdm(total=NTRIALS)
num_skipped = 0
i=0
while i < NTRIALS:
    model = DGCA(num_states=seed_graph.num_states)
    runner.reset()
    final_graph = runner.run(model,seed_graph)
    final_graph_nosl = final_graph.no_selfloops()
    ok = check_conditions(final_graph, conditions, VERBOSE)
    if ok:
        (motifs, 
        zscores13, 
        gcount13, 
        avgcount13, 
        stdcount13) = gt.motif_significance(final_graph_nosl.to_gt(), 
                                        k=3, motif_list=motifs13, 
                                        self_loops=False, n_shuffles=N_SHUFFLES,
                                        full_output=True, p=RAND_ESU)
        this_result = [zscores13, gcount13, avgcount13, stdcount13]
        this_result.extend([model.encode(), final_graph, *runner.attractor_info()])
        results.append(this_result)
        i += 1
        pbar.update(1)
    else:
        num_skipped += 1
        continue

print(f'Number skipped: {num_skipped}')
# save results
timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
results_fn = os.path.join(PROJ_DIR,'results', f'results_{timestr}.p')
print(f'Saving results to: {results_fn}')
with open(results_fn, 'wb') as res_handle:
    pickle.dump(results, res_handle, protocol=pickle.HIGHEST_PROTOCOL)
