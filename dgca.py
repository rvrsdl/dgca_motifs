"""
Simple implementation of the DGCA system
"""
#%%
from __future__ import annotations
from dataclasses import dataclass
from multiset import FrozenMultiset
import numpy as np
import graph_tool.all as gt
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%

def rindex(it, li) -> int:
    """
    Reverse index ie. the
    index of last occurence of item in list
    """
    return len(li) - 1 - li[::-1].index(it)


@dataclass
class GraphDef:
    A: np.ndarray # n*n
    S: np.ndarray # s*n
    num_states: int = 0

    def __post_init__(self) -> None:
        """
        Just check variables are ok after initialisation.
        And change to smaller dtypes
        """
        assert self.A.shape[0]==self.A.shape[1], "Adjacency matrix must be square"
        assert self.A.shape[1]==self.S.shape[1], "Adjacency and State matrix sizes don't match"
        self.num_states = self.S.shape[0]
        self.A = self.A.astype(np.bool_)
        self.S = self.S.astype(np.bool_)

    def __str__(self) -> str:
        return f"Graph with {self.size()} nodes and {self.num_edges()} edges"

    def size(self) -> int:
        return self.A.shape[0]
    
    def num_edges(self) -> int:
        return np.sum(self.A)

    def connectivity(self) -> float:
        if self.size()>0:
            return self.num_edges() / self.size()**2
        else:
            return 0 # or could do np.nan?
        
    def state_hash(self) -> int:
        """
        Returns a hash of all the nieghbourhood state info.
        This is good as a preliminary isomorphism check (if
        two hashes are not the same then the graphs are definitely
        different).
        """
        info = DGCA.gather_neighbourhood_info(self, bidirectional=True, renorm=False)
        return hash(FrozenMultiset(map(tuple, info.T.tolist())))
    
    def to_edgelist(self) -> np.ndarray:
        """
        Returns an E*3 numpy array in the format
        source, target, weight
        source, target, weight
        etc.
        (in this case weights are all 1)
        """
        es = np.nonzero(self.A)
        edge_list = np.array([es[0], es[1], self.A[es]]).T
        return edge_list
    
    def states_1d(self) -> list[int]:
        """
        Converts states out of one-hot encoding into a list of ints.
        eg. [[1,0,0,0],[0,1,1,0],[0,0,0,1]] -> [0,1,1,2]
        """
        return np.argmax(self.S, axis=0).tolist()

    def to_gt(self, basic: bool = False) -> gt.Graph:
        """
        Converts it to a graph-tool graph.
        Good for visualisation and isomorphism checks.
        Nodes are coloured by state.
        Use basic=True if you just want the graph structure
        and don't care about states, node colours etc.
        """
        g = gt.Graph(self.size())
        g.add_edge_list(self.to_edgelist())
        if basic:
            return g
        states = g.new_vertex_property('int', self.states_1d())
        g.vp['state'] = states # make into an "internal" property
        # Set node colours for plotting
        # Same colours as same as in plot_space_time
        states_1d = self.states_1d()
        cmap = plt.get_cmap('viridis', self.num_states+1)
        state_colours = cmap(states_1d)
        g.vp['plot_colour'] =  g.new_vertex_property('vector<double>', state_colours)
        return g    
    
    def is_isomorphic(self, other: GraphDef) -> bool:
        """
        Checks if this graph is isomorhpic with another, conditional on node states.
        """
        ne1 = self.num_edges()
        ne2 = other.num_edges()
        if ne1!=ne2:
            return False
        if ne1==0:
            # Neither have any edges: structure doesn't matter so just check states match
            s1 = self.states_1d()
            s2 = other.states_1d()
            s1.sort()
            s2.sort()
            return s1==s2 
        # Use graph-tool to check for isomorphism
        gt1, gt2 = self.to_gt(), other.to_gt()
        # Despite the function name, if you set subgraph=False, it does whole graph isomorphism
        # The vertex_label param is used to condition the isomorphism on node state.
        mapping = gt.subgraph_isomorphism(gt1, gt2, 
                                vertex_label=[gt1.vp.state, gt2.vp.state],
                                subgraph=False)
        #TODO: timeout
        # is isomorphic if at least one mapping was found
        return False if len(mapping)==0 else True
        
    def draw_gt(self, draw_edge_wgt: bool = False,
                 pos: gt.VertexPropertyMap | None = None,
                 interactive: bool = False,
                 **kwargs) -> gt.VertexPropertyMap:
        """
        Draws a the graph using the graph-tool library 
        Relies on node and edge properties set by to_gt()
        Returns the node positions, which can then be passed in at the next
        call so that original nodes don't move to much if you are adding more etc.
        NB use output=filename to write to a file.
        """
        if self.size()==0:
            print('Empty graph - can''t draw')
            return None
            #TODO: will this break if called inside a plotting routine?
        g = self.to_gt(pos=pos)
        if draw_edge_wgt:
            edge_pen_width = gt.prop_to_size(g.ep.abswgt,mi=1,ma=7)
        else:
            edge_pen_width = None
        if interactive:
            pos_out = gt.interactive_window(g, pos=g.vp['pos'], vertex_fill_color=g.vp.plot_colour, edge_pen_width=edge_pen_width, **kwargs)
        else:
            pos_out = gt.graph_draw(g, pos=g.vp['pos'], vertex_fill_color=g.vp.plot_colour, edge_pen_width=edge_pen_width, **kwargs)
            return pos_out
        
    def no_selfloops(self) -> GraphDef:
        """
        Returns a copy of the graph in which all self-loops have been removed
        """
        out_A = self.A.copy()
        # set values on the diagonal to zero
        out_A[np.eye(out_A.shape[0], dtype=np.bool_)] = 0 
        return GraphDef(out_A, S.copy(), self.num_states)
    

class DGCA:
    """
    Holds the SLP weights and other params, but can be run with any seed graph.
    """

    def __init__(self, num_states: int,
                 action_info_bidirectional: bool = True,
                 action_info_renorm: bool = True,
                 state_info_bidirectional: bool = True,
                 state_info_renorm: bool = True) -> None:
        self.num_states = num_states
        self.action_info_bidirectional = action_info_bidirectional
        self.action_info_renorm = action_info_renorm
        self.state_info_bidirectional = state_info_bidirectional
        self.state_info_renorm = state_info_renorm
        action_slp_inp_size = 3*num_states+1 if action_info_bidirectional else 2*num_states+1
        state_slp_inp_size = 3*num_states+1 if state_info_bidirectional else 2*num_states+1
        self.action_slp_wgts = np.random.uniform(low=-1, high=1, size=(15, action_slp_inp_size))
        self.state_slp_wgts = np.random.uniform(low=-1, high=1, size=(self.num_states, state_slp_inp_size))


    @staticmethod
    def gather_neighbourhood_info(G: GraphDef,
                                bidirectional: bool, renorm: bool) -> np.ndarray:
        A, S = G.A, G.S
        out = np.vstack((S, S @ A))
        if bidirectional:
            out = np.vstack((out, S @ A.T))
        # Append bias row of ones
        out = np.vstack((out, np.ones(S.shape[1])))
        if renorm:
            # Do this after appending bias to avoid dividing by zero
            out = out / np.max(np.abs(out), axis=0, keepdims=True)
            # Reset bias to 1 (??)
            #out[-1,:] = np.ones(S.shape[1])
        return out
    
    def action_update(self, G: GraphDef) -> tuple[np.ndarray, np.ndarray]:
        C = self.gather_neighbourhood_info(G, self.action_info_bidirectional, self.action_info_renorm)
        D = self.action_slp_wgts @ C
        # Interpret output
        K = np.vstack((
            D[0:3,:] == np.max(D[0:3,:], axis=0, keepdims=True),
            D[3:7,:] == np.max(D[3:7,:], axis=0, keepdims=True),
            D[7:11,:] == np.max(D[7:11,:], axis=0, keepdims=True),
            D[11:15,:] == np.max(D[11:15,:], axis=0, keepdims=True),
        ))
        # - action choices
        remove = K[1,:]
        stay = K[2,:]
        divide = K[3,:]
        keep = np.hstack((np.logical_not(remove), divide))
        # - new node wiring
        k_fi, k_fa, k_ft = K[4, :], K[5, :], K[6, :]
        k_bi, k_ba, k_bt = K[8, :], K[9, :], K[10,:]
        k_ni, k_na, k_nt = K[12,:], K[13,:], K[14,:]
        # Restructure adjacency matrix
        I = np.eye(G.size())
        Qm = np.array([[1,0],[0,0]])
        Qf = np.array([[0,1],[0,0]])
        Qb = np.array([[0,0],[1,0]])
        Qn = np.array([[0,0],[0,1]])
        A, S = G.A, G.S
        A_new = np.kron(Qm, A) \
            + np.kron(Qf, (I*np.diag(k_fi) + A*np.diag(k_fa) + A.T*np.diag(k_ft))) \
            + np.kron(Qb, (I*np.diag(k_bi) + A*np.diag(k_ba) + A.T*np.diag(k_bt))) \
            + np.kron(Qn, (I*np.diag(k_ni) + A*np.diag(k_na) + A.T*np.diag(k_nt)));  
        # keep only the nodes we need
        A_new = A_new[keep,:][:,keep]
        # Duplicate relevant cols of state matrix
        S_new = np.hstack((S, S))
        S_new = S_new[:,keep]
        return GraphDef(A_new, S_new)

    def state_update(self, G: GraphDef) -> tuple[np.ndarray, np.ndarray]:
        C = self.gather_neighbourhood_info(G, self.state_info_bidirectional, self.state_info_renorm)
        D = self.state_slp_wgts @ C
        # Make one-hot
        S_new = D==np.max(D, axis=0, keepdims=True)
        return GraphDef(G.A, S_new)

    def step(self, G: GraphDef) -> tuple[np.ndarray, np.ndarray]:
        """
        Simply performs an action update followed by a state update
        """
        return self.state_update(self.action_update(G))
    

class Runner:
    """
    To hold the run params and check if we have entered an attractor
    (so can stop early).
    """

    def __init__(self, max_steps: int, max_size: int) -> None:
        self.max_steps = max_steps
        self.max_size = max_size
        self.graphs: list[GraphDef] = []
        self.hashes: list[int] = []
        self.status = 'ready'
    
    def record(self, G: GraphDef) -> None:
        """
        Adds the graph and its hash to the records.
        """
        self.graphs.append(G)
        self.hashes.append(G.state_hash())

    def reset(self):
        self.graphs: list[GraphDef] = []
        self.hashes: list[int] = []
        self.status = 'ready'

    def already_seen(self, G: GraphDef) -> None:
        this_hash = G.state_hash()
        if this_hash in self.hashes:
            possible_match = self.graphs[self.hashes.index(this_hash)]
            iso = G.is_isomorphic(possible_match)
            if not iso:
                # The quick hash check gave a false positive.
                # TODO: change the hash
                pass
            return iso
        else:
            return False

    def run(self, dgca: DGCA, seed_graph: GraphDef) -> GraphDef:
        """
        Runs for the full number of steps or stops early if the
        graph becomes too big or the system enters an attractor.
        """
        current_graph = seed_graph
        self.record(current_graph)
        for i in range(self.max_steps):
            next_graph = dgca.step(current_graph)
            if next_graph.size() == 0:
                self.status = 'zero_nodes'
                self.record(next_graph)
                return next_graph
            if next_graph.size() > self.max_size:
                self.status = 'max_size'
                self.record(next_graph)
                return current_graph # because next_graph is too big
            if self.already_seen(next_graph):
                # We are in an attractor so no need to go further
                # Add it to the record anyway so that we can spot the cycle
                self.status = 'attractor'
                self.record(next_graph)
                return next_graph
            self.record(next_graph)
            current_graph = next_graph
        self.status = 'max_steps'
        return current_graph

    def graph_size(self) -> list[int]:
        """
        Returns a list of the size of the graph at each step
        of the run.
        """
        if self.status == 'ready':
            print("Hasn't been run yet!")
        return [g.size() for g in self.graphs]
    
    def graph_connectivity(self) -> list[float]:
        if self.status == 'ready':
            print("Hasn't been run yet!")
        return [g.connectivity() for g in self.graphs]
    
    def attractor_info(self) -> tuple[int, int]:
        """
        Returns the transient and attractor length.
        An attractor length of zero means no attractor
        was found.
        """
        if self.status == 'ready':
            print("Hasn't been run yet!")
        if self.hashes[-1] in self.hashes[:-1]:
            ind = rindex(self.hashes[-1], self.hashes[:-1])
            attr_len = len(self.hashes) - 1 - ind
        else:
            attr_len = 0
        trans_len = len(self.hashes) - attr_len
        return trans_len, attr_len
# %%
for i in tqdm(range(1000)):
    dgca = DGCA(num_states=3)
    seed = GraphDef(np.array([[1,1,0],[0,0,1],[1,0,0]]), np.array([[1,0,0],[0,1,0],[0,1,0]]).T)
    runner = Runner(max_steps=100, max_size=500)
    out = runner.run(dgca, seed)
    if runner.status=='attractor' and out.size()>100:
        break
print(out)
print(f"{runner.status} after {len(runner.hashes)} steps")
#%%#
g = out.to_gt()
gt.remove_self_loops(g)
gt.graph_draw(g, vertex_fill_color=g.vp.plot_colour)
# 1, 2, 3, 2f, 4, 5, 2

# %%
g.save('blah4.dot')
# %%
