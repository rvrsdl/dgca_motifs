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
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
from wrapt_timeout_decorator.wrapt_timeout_decorator import timeout
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
#%%

def rindex(it, li) -> int:
    """
    Reverse index() ie.
    index of last occurence of item in list
    """
    return len(li) - 1 - li[::-1].index(it)

def onehot(x: np.ndarray) -> np.ndarray:
    """
    One-hot helper function. Turns an array of floats into a binary array with 
    a single 1 per row (in the location of the highest valued float of each row).
    """
    tf = x == np.max(x, axis=1, keepdims=True)
    return tf.astype(int)

@dataclass
class GraphDef:
    """
    Just holds the basic info about a single graph:
    - A: its adjacency matrix. dimensions: (n,n)
    - S: its states matri - each row is a one-hot representation 
         of that node's state. dimensions: (n*s)
    Also has various utility methods.
    - draw_gt() is useful for displaying the graph using the graph-tool
      librar's drawing functionality.
    """

    A: np.ndarray # n*n
    S: np.ndarray # n*s
    num_states: int = 0

    def __post_init__(self) -> None:
        """
        Just check variables are ok after initialisation.
        And change to smaller dtypes
        """
        assert self.A.shape[0]==self.A.shape[1], "Adjacency matrix must be square"
        assert self.A.shape[0]==self.S.shape[0], "Adjacency and State matrix sizes don't match"
        self.num_states = self.S.shape[1]
        # self.A = self.A.astype(np.bool_)
        # self.S = self.S.astype(np.bool_)

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
        
    def get_components(self) -> tuple[np.ndarray]:
        """
        Returns a number for each node indicating which component
        it is part of.
        eg. [1,1,2,2,1] means nodes 0,1,4 form one connected component
        and nodes 2&3 form another.
        """
        # use scipy.sparse.csgraph.connected_components to identify connected
        # components. Treat graph as undirected because we don't care which 
        # way the edges are going for theses purposes.
        _, cc = connected_components(self.A, directed=False)
        # count number of nodes in each component (will be sorted by component label 0->n_components)
        _, counts = np.unique(cc, return_counts=True)
        # TODO: would it be better to use graph_tool.topology.label_components
        # (fewer dependencies, but required converting to gt.Graph first.)
        return cc, counts
    
    def get_largest_component_frac(self) -> float:
        """
        Returns the size of the largest component as a fraction of 
        the total number of nodes in the graph.
        """
        if self.size()==0:
            return 0
        else:
            _, component_sizes = self.get_components()
            return np.max(component_sizes) / self.size()

    def state_hash(self) -> int:
        """
        Returns a hash of all the nieghbourhood state info.
        This is good as a preliminary isomorphism check (if
        two hashes are not the same then the graphs are definitely
        different).
        """
        info = DGCA.gather_neighbourhood_info(self, bidirectional=True, renorm=False)
        return hash(FrozenMultiset(map(tuple, info.tolist())))
    
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
        eg. [[1,0,0,0],[0,0,1,0],[0,0,0,1]] -> [0,2,3]
        """
        return np.argmax(self.S, axis=1).tolist()
    

    def to_gt(self, basic: bool = False,
              pos: gt.VertexPropertyMap | None = None) -> gt.Graph:
        """
        Converts it to a graph-tool graph.
        Good for visualisation and isomorphism checks.
        Nodes are coloured by state.
        Use basic=True if you just want the graph structure
        and don't care about states, node colours etc.
        """
        num_nodes = self.size()
        #es = np.nonzero(self.graph_mat)
        #edge_list = np.array([es[0], es[1], self.graph_mat[es], np.abs(self.graph_mat[es])]).T
        edge_list = self.to_edgelist()
        g = gt.Graph(num_nodes)
        g.add_edge_list(edge_list, eprops=[("wgt","double")])
        if not basic:
            states = g.new_vertex_property('int', self.states_1d())
            g.vp['state'] = states # make into an "internal" property
            # Set node colours for plotting
            # Same colours as same as in plot_space_time
            states_1d = self.states_1d()
            cmap = plt.get_cmap('viridis', self.num_states+1)
            state_colours = cmap(states_1d)
            g.vp['plot_colour'] =  g.new_vertex_property('vector<double>', state_colours)
            g.vp['pos'] = gt.sfdp_layout(g,pos=pos)
        return g
    
    @timeout(5, use_signals=False)
    def is_isomorphic(self, other: GraphDef) -> bool:
        """
        Checks if this graph is isomorhpic with another, conditional on node states.
        The decorator makes this function raise a timeout error if it takes longer than 5 seconds.
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
        return GraphDef(out_A, self.S.copy(), self.num_states)
    

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
        self.action_slp_wgts = np.random.uniform(low=-1, high=1, size=(action_slp_inp_size, 15))
        self.state_slp_wgts = np.random.uniform(low=-1, high=1, size=(state_slp_inp_size, self.num_states))


    @staticmethod
    def gather_neighbourhood_info(G: GraphDef,
                                bidirectional: bool, renorm: bool) -> np.ndarray:
        A, S = G.A, G.S
        # Equations #4 and #5 in paper
        out = np.hstack((S, A @ S)) # dims: (n,2s)
        if bidirectional:
            out = np.hstack((out, A.T @ S)) # dims: (n,3s)
        # Append bias col of ones
        out = np.hstack((out, np.ones((S.shape[0],1))))
        if renorm:
            # Do this after appending bias to avoid dividing by zero
            out = out / np.max(np.abs(out), axis=1, keepdims=True)
            # Reset bias to 1 (??)
            #out[:,-1] = np.ones((S.shape[0],1))
        return out
    
    def action_update(self, G: GraphDef) -> tuple[np.ndarray, np.ndarray]:
        C = self.gather_neighbourhood_info(G, self.action_info_bidirectional, self.action_info_renorm)
        D = C @ self.action_slp_wgts # dims: (n,3s+1) @ (3s+1, 15) = (n,15)
        # Interpret output
        # - one hot in sections
        K = np.hstack((onehot(D[:,0:3]), onehot(D[:,3:7]), onehot(D[:,7:11]), onehot(D[:,11:15])))
        K = K.T # transpose to match Table 1 in paper, dims: (15, n)
        # - action choices
        remove = K[0,:]
        noaction = K[1,:]
        divide = K[2,:]
        keep = np.hstack((np.logical_not(remove), divide)).astype(bool)
        # - new node wiring
        k_f0, k_fi, k_fa, k_ft = K[3, :], K[4, :], K[5, :], K[6, :]
        k_b0, k_bi, k_ba, k_bt = K[7, :], K[8, :], K[9, :], K[10,:]
        k_n0, k_ni, k_na, k_nt = K[11,:], K[12,:], K[13,:], K[14,:]
        # Restructure adjacency matrix
        I = np.eye(G.size())
        Qm = np.array([[1,0],[0,0]])
        Qf = np.array([[0,1],[0,0]])
        Qb = np.array([[0,0],[1,0]])
        Qn = np.array([[0,0],[0,1]])
        A, S = G.A, G.S
        A_new = np.kron(Qm, A) \
            + np.kron(Qf, (I @ np.diag(k_fi) + A @ np.diag(k_fa) + A.T @ np.diag(k_ft))) \
            + np.kron(Qb, (np.diag(k_bi) @ I + np.diag(k_ba) @ A + np.diag(k_bt) @ A.T)) \
            + np.kron(Qn, (np.diag(k_ni) @ I + np.diag(k_na) @ A + np.diag(k_nt) @ A.T))
        # keep only the nodes we need
        A_new = A_new[keep,:][:,keep] # logical indexing to keep only the row/cols we need.
        # Duplicate relevant cols of state matrix
        S_new = np.vstack((S, S))
        S_new = S_new[keep,:]
        return GraphDef(A_new, S_new)

    def state_update(self, G: GraphDef) -> tuple[np.ndarray, np.ndarray]:
        C = self.gather_neighbourhood_info(G, self.state_info_bidirectional, self.state_info_renorm)
        D = C @ self.state_slp_wgts #dims: (n,3s+1) @ (3s+1, s) = (n,s)
        # Make one-hot
        S_new = onehot(D)
        return GraphDef(G.A, S_new)

    def step(self, G: GraphDef) -> tuple[np.ndarray, np.ndarray]:
        """
        Simply performs an action update followed by a state update
        """
        return self.state_update(self.action_update(G))
    
    def encode(self) -> str:
        """
        Serialises the model using jsonpickle
        """
        return jsonpickle.encode(self)
    

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
        self.ids: list[int] = []
        self.status = 'ready'
        self.hash_offset = 1
    
    def record(self, G: GraphDef, duplicate_of: int | None = None) -> None:
        """
        Adds the graph and its hash to the records.
        """
        # Figure out the id number
        if len(self.ids)==0:
            self.ids.append(0)
        elif duplicate_of or duplicate_of==0:
            # This graph is isomorphic to one we have already seen
            self.ids.append(self.ids[duplicate_of])
        else: 
            self.ids.append(max(self.ids)+1)
        self.graphs.append(G)
        self.hashes.append(G.state_hash())
        

    def reset(self):
        self.graphs: list[GraphDef] = []
        self.hashes: list[int] = []
        self.ids: list[int] = []
        self.status = 'ready'

    def already_seen(self, G: GraphDef) -> tuple[bool,bool]:
        """
        Checks if we have already seen this graph
        """
        this_hash = G.state_hash()
        if this_hash in self.hashes:
            match_idx = [i for i,x in enumerate(self.hashes) if x==this_hash]
            iso_tf = []
            for m in match_idx:
                try:
                    is_iso = G.is_isomorphic(self.graphs[m])
                    if not is_iso:
                        # We got a false positive from the hash comparison so rehash the one in the archive
                        # self.hashes[m] = hash(self.hashes[m]+  self.hash_offset) # get a new hash value by hashing the original hash value!
                        # self.hash_offset += 1
                        pass
                    iso_tf.append(is_iso)
                except TimeoutError:
                    # Isomorphism check timed out. 
                    # Assume it IS isomorphic to be safe
                    iso_tf.append(True)
                    print('Warning: isomorphism check timed out')
            any_iso = any(iso_tf)
            if any_iso:
                idx = match_idx[iso_tf.index(True)]
            else:
                idx = None
            return any_iso, idx
        else:
            return False, None

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
            already_seen, match_idx = self.already_seen(next_graph)
            if already_seen:
                # We are in an attractor so no need to go further
                # Add it to the record anyway so that we can spot the cycle
                self.status = 'attractor'
                self.record(next_graph, duplicate_of=match_idx)
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
        if self.ids[-1] in self.ids[:-1]:
            ind = rindex(self.ids[-1], self.ids[:-1])
            attr_len = len(self.ids) - 1 - ind
        else:
            attr_len = 0
        trans_len = len(self.ids) - attr_len - 1
        return trans_len, attr_len
# # %%
# for i in tqdm(range(1000)):
#     seed = GraphDef(np.array([[1,1,0],[0,0,1],[1,0,0]]), np.array([[1,0],[0,1],[0,1]]))
#     dgca = DGCA(num_states=seed.num_states)
#     runner = Runner(max_steps=100, max_size=500)
#     out = runner.run(dgca, seed)
#     if runner.status=='attractor' and out.size()>100:
#         break
# print(out)
# print(f"{runner.status} after {len(runner.hashes)} steps")
# #%%#
# g = out.to_gt()
# gt.remove_self_loops(g)
# gt.graph_draw(g, vertex_fill_color=g.vp.plot_colour)
# # 1, 2, 3, 2f, 4, 5, 2

# # %%
# g.save('blah4.dot')
# %%
if __name__=='__main__':
    print('Demo')
    g0 = GraphDef(np.array([[1,1,0],[0,0,1],[1,0,0]]), np.array([[1,0],[0,1],[0,1]]))
    print('Seed graph (step 0):')
    g0.draw_gt()
    dgca = DGCA(num_states=g0.num_states)
    g1 = dgca.step(g0)
    print('Step 1:')
    g1.draw_gt()
    g2 = dgca.step(g1)
    print('Step 2:')
    g2.draw_gt()
# %%
