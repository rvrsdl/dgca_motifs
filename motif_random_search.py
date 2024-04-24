import numpy as np
import graph_tool.all as gt
from dgca import GraphDef, DGCA, Runner

dgca = DGCA(num_states=3)
seed = GraphDef(np.array([[1]]), np.array([[1,0,0]]).T)
runner = Runner(dgca, seed, max_steps=10, max_size=100)
grown_graph = runner.run()
gt.motif_significance(grown_graph.to_gt(basic=True), k=3, p=[1,1,0.5])
