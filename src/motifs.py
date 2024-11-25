"""
Functions to count motif occurrence and significance
using graph-tool. Uses v4 GraphDef
Actually gt.motif_significance() does it all for us, so no
need to write extra funcs.
This can just contain plotting tools
"""
#%%
import itertools
import matplotlib.pyplot as plt
#import zstandard
import graph_tool.all as gt
import numpy as np
##%%
def generate_motifs(k: int, self_loops: bool = False) -> list[gt.Graph]:
    """
    Generates all motifs for a given number of nodes.
    By default self-loops are not allowed
    For 3-node connected motifs without self-loops it should return 
    the 13 graphs shown on p.42 of Uri Alon, *An Introduction to Systems
    Biology*
    The 199 4-node connected motifs are shown on p.82.
    n_nodes | n_motifs | with selfloops
    --------|----------|---------------
    2       |2         |7
    3       |13        |86
    4       |199       |2818
    """
    nodes = range(k)
    if self_loops:
        all_edges = list(itertools.product(nodes, nodes))
    else:
        all_edges = list(itertools.permutations(nodes, 2))
    #if connected:
    min_edges = k - 1
    #else:
    #    min_edges = 0
    n_edges = range(min_edges, len(all_edges)+1)
    all_gr = []
    for ne in n_edges:
        # find all graphs with this number of edges
        edge_combos = itertools.combinations(all_edges, ne)
        gr_ne_ok = [] # to hold valid graphs with this number of edges
        for edge_list in edge_combos:
            gg = gt.Graph(k)
            gg.add_edge_list(edge_list)
            _, h = gt.label_components(gg, directed=False)
            is_connected = h.shape[0]==1 # only one component
            #nx.is_weakly_connected(gg) # ignore edge direction
            if not is_connected: 
                continue
            if len(gr_ne_ok)==0:
                # add the first one to the list
                gr_ne_ok.append(gg)
            else:
                # check that this graph isn't an isomorphism of any we already have.
                #iso = any(map(lambda x: nx.is_isomorphic(gg,x), gr_ne_ok))
                iso = map(lambda x: gt.isomorphism(gg,x), gr_ne_ok)
                if not any(iso):
                    gr_ne_ok.append(gg)
        all_gr.extend(gr_ne_ok)
    return all_gr

def generate_named_triads() -> dict[str, gt.Graph]:
    """
    Hardcoded version of generate_motifs(3, self_loops=False)
    Returns a dict of names to motifs (in simplest to most complex order)
    Names are from: [[@shellmanMetabolicNetworkMotifs2014]]
    But ordering is that used by graph-tool 
    (sorted according to in-degree sequence, out-degree-sequence, 
    and number of edges (in this order).)
    NB dict order guaranteed in Python 3.7+: 
    https://mail.python.org/pipermail/python-dev/2017-December/151283.html
    """
    out = dict()
    m0 = gt.Graph(3)
    m0.add_edge_list([(1,0),(1,2)])
    out['V-Out'] = m0
    m1 = gt.Graph(3)
    m1.add_edge_list([(0,1),(2,1)])
    out['V-In'] = m1
    m2 = gt.Graph(3)
    m2.add_edge_list([(0,1),(1,2)])
    out['3-Chain'] = m2
    m3 = gt.Graph(3)
    m3.add_edge_list([(0,1),(0,2),(2,1)])
    out['Feed Forward'] = m3
    m4 = gt.Graph(3)
    m4.add_edge_list([(0,1),(1,0),(1,2)])
    out['Mutual Out'] = m4
    m5 = gt.Graph(3)
    m5.add_edge_list([(0,1),(1,0),(2,1)])
    out['Mutual In'] = m5
    m6 = gt.Graph(3)
    m6.add_edge_list([(0,1),(1,2),(2,0)])
    out['3-Loop'] = m6
    m7 = gt.Graph(3)
    m7.add_edge_list([(0,1),(1,0),(0,2),(1,2)])
    out['Regulating Mutual'] = m7
    m8 = gt.Graph(3)
    m8.add_edge_list([(0,1),(1,0),(2,0),(2,1)])
    out['Regulated Mutual'] = m8
    m9 = gt.Graph(3)
    m9.add_edge_list([(0,1),(1,0),(1,2),(2,1)])
    out['Mutual V'] = m9
    m10 = gt.Graph(3)
    m10.add_edge_list([(0,1),(1,0),(1,2),(2,0)])
    out['Mutual and 3-Chain'] = m10
    m11 = gt.Graph(3)
    m11.add_edge_list([(0,1),(1,0),(0,2),(2,0),(1,2)])
    out['Semi-Clique'] = m11
    m12 = gt.Graph(3)
    m12.add_edge_list([(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)])
    out['Clique'] = m12
    return out

# Need to do this for the below to work - not sure when
# plt.switch_backend("GTK3Cairo")
def ticklabels_graphs(ax: plt.Axes, to_plot: list[gt.Graph], 
                      xy: str, vpos: gt.VertexPropertyMap | None = None, 
                      **kwargs) -> None:
    """
    Set the X and/or Y tick labels to graphs.
    Good for plotting motif counts etc.
    """
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    xrange = abs(xl[1] - xl[0])
    yrange = abs(yl[1] - yl[0])
    for i,g in enumerate(to_plot):
        if 'x' in xy:
            #left, bottom, width, height
            xlab_ax = ax.inset_axes([i*(1/xrange), -1/xrange, 1/xrange, 1/xrange])
            xlab_ax.set_axis_off()
            gt.graph_draw(g, pos=vpos, mplfig=xlab_ax, **kwargs)
        if 'y' in xy:
            ylab_ax = ax.inset_axes([-1/xrange, i*(1/xrange), 1/xrange, 1/xrange])
            ylab_ax.set_axis_off()
            gt.graph_draw(g, pos=vpos, mplfig=ylab_ax, **kwargs)
    if 'x' in xy:
        ax.set_xticklabels([]) # remove original xtick_labels
    if 'y' in xy:
        ax.set_yticklabels([]) # remove original xtick_labels

def xticklabels_13motifs(ax, fit_view: bool = True, ink_scale: float = 1.5, **kwargs) -> None:
    """
    Wrapper around the above seting lots of the params for us.
    Set gt_order=True to get these back in the order 
    that gt_motif_significance outputs them!
    NB Now in gt order anyway!
    """
    m13 = list(generate_named_triads().values())
    # if gt_order:
    #     m13 = gt_order_motifs(3, motifs=m13)
    pmap = m13[0].new_vertex_property("vector<double>")
    pmap.set_2d_array(np.array([[0,1,2],[1,0,1]]))
    #plt.switch_backend("GTK3Cairo")
    ticklabels_graphs(ax, m13, 'x',vpos=pmap, fit_view=fit_view, ink_scale=ink_scale, **kwargs)

def gt_order(motifs) -> list[gt.Graph]:
    g = gt.random_graph(100, lambda: (3,3))
    ordered_motifs, _ = gt.motif_significance(g, 3, motif_list=motifs)
    return ordered_motifs

def get_natural_conc(names: list[str] = ["ecoli_transcription/v1.0", "yeast_transcription","celegansneural"],
                     self_loops: bool = False, n_shuffles: int = 100,
                     rand_esu: list[float] = [0.9,0.8,0.7]) -> np.array:
    m3 = generate_named_triads()
    m3l = list(m3.values())
    conc = dict()
    for name in names:
        print(f'Processing {name}')
        gr = gt.collection.ns[name]
        if not self_loops:
            gt.remove_self_loops(gr)
        sig = gt.motif_significance(gr, k=3, 
                                    motif_list=m3l, full_output=True,
                                    n_shuffles=n_shuffles,
                                    p=rand_esu, self_loops=self_loops)
        conc[name] = np.array(sig[2]) / np.sum(sig[2])
    return conc

def gt_order_motifs(k: int, motifs: list[gt.Graph]) -> list[gt.Graph]:
    g = gt.random_graph(100, lambda: (3,3))
    m_out, _= gt.motif_significance(g, k, motif_list=motifs)
    return m_out


def plot13motifs(order: str='graph-tool', nrows: int = 2):
    """
    For the paper I did:
    f = plot13motifs(order='milo',nrows=1)
    f.savefig('13TriadsMiloOrder2.pdf')
    """
    m13dict = generate_named_triads()
    pmap = m13dict['V-Out'].new_vertex_property("vector<double>")
    pmap.set_2d_array(np.array([[0,1,2],[1,0,1]]))
    plt.switch_backend("GTK3Cairo")
    ncols = -(13//-nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2,nrows*2), constrained_layout=True)
    axs = axs.flatten()
    for i, k in enumerate(motif_sort(order)): #g, a in zip(m13dict.keys(), axs):
        gt.graph_draw(m13dict[k], pos=pmap, mplfig=axs[i])
        title_y = 1 #1 if i<7 else -0.2
        if len(k)<=12:
            triad_name= k
        else:
            triad_name = k.replace(' ','\n',1)
        axs[i].set_title(f'{i+1}: {triad_name}', y=title_y, fontsize=22)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_axis_off()
    for j in range(i+1,len(axs)):
        axs[j].remove()
    #fig.suptitle('The 13 Triads')
    print(f'Triads shown in {order} order')
    #plt.show(block=True)
    return fig

def motif_sort(order: str):
    """
    Returns the motif names in the order used by different publications:
    - graph-tool
    - milo @miloSuperfamiliesEvolvedDesigned2004, @shellmanMetabolicNetworkMotifs2014
    - eom @eomExploringLocalStructural2006
    """
    if order in ('graph-tool','gt'):
        out = ['V-Out', 'V-In', '3-Chain', 'Feed Forward', 
               'Mutual Out', 'Mutual In', '3-Loop', 'Regulating Mutual', 
               'Regulated Mutual', 'Mutual V', 'Mutual and 3-Chain', 
               'Semi-Clique', 'Clique']
    elif order in ('milo','shellman'):
        out = ['V-Out', 'V-In', '3-Chain', 'Mutual In', 'Mutual Out',
                'Mutual V', 'Feed Forward','3-Loop', 'Regulated Mutual',
                'Regulating Mutual', 'Mutual and 3-Chain', 
               'Semi-Clique', 'Clique']
    elif order in ('eom'):
        out = ['V-Out', '3-Chain', 'Mutual Out', 'V-In', 'Feed Forward',
               'Regulating Mutual','Mutual In', 'Mutual V', '3-Loop', 
               'Mutual and 3-Chain', 'Regulated Mutual',
               'Semi-Clique', 'Clique'] 
    else:
        raise ValueError('Unrecosgnised value for order param')
    return out

def gt2milo_sortix() -> list[int]:
    """
    Returns a sortix for converting gt motif order to Milo order
    """
    return [motif_sort('gt').index(k) for k in motif_sort('milo')]

target_tsp = {
    'ecoli': np.array([-0.613, -0.377, -0.313, 0.620, -0.010, -0.015, -0.001, -0.001, -0.000, 0.000, 0.000, 0.000, 0.000]),
    'yeast': np.array([-0.561, -0.453, -0.418, 0.537, -0.008, 0.121, -0.001, 0.040, -0.000, 0.000, 0.040, 0.000, 0.000]),
    'celegans': np.array([-0.341, -0.364, -0.230, 0.001, 0.145, 0.100, -0.151, 0.552, 0.367, 0.179, 0.079, 0.316, 0.259])
}
# %%
# # FOr paper
# names = ['V-Out',
#  'V-In',
#  '3-Chain',
#  'Feed\nForward',
#  'Mutual\nOut',
#  'Mutual\nIn',
#  '3-Loop',
#  'Regulating\nMutual',
#  'Regulated\nMutual',
#  'Mutual V',
#  'Mutual\nand 3-Chain',
#  'Semi-\nClique',
#  'Clique']
# m13dict = generate_named_3motifs()
# pmap = m13dict['V-Out'].new_vertex_property("vector<double>")
# pmap.set_2d_array(np.array([[0,1,2],[1,0,1]]))
# plt.switch_backend("GTK3Cairo")
# fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(5,6), constrained_layout=True)
# axs = axs.flatten()
# for i, k in enumerate(m13dict): #g, a in zip(m13dict.keys(), axs):
#     gt.graph_draw(m13dict[k], pos=pmap, mplfig=axs[i])
#     title_y = 1
#     axs[i].set_title(f'{i+1}: {names[i]}', y=title_y)
#     axs[i].set_xticks([])
#     axs[i].set_yticks([])
#     axs[i].axis('off')
# axs[-1].remove()
# axs[-2].remove()
# axs[-3].remove()
# #fig.suptitle('13 Triads (in graph-tool order)')
# fig.savefig('13Triads7.pdf')

# %%

# # example with natural networks:
# ecoli = gt.collection.ns["ecoli_transcription/v1.1"]
# yeast = gt.collection.ns["yeast_transcription"]
# celegans = gt.collection.ns["celegansneural"]
# tadpole = gt.collection.ns["cintestinalis"]
# gt.remove_self_loops(ecoli)
# gt.remove_self_loops(yeast)
# gt.remove_self_loops(celegans)
# gt.remove_self_loops(tadpole)
# m13 = list(generate_named_triads().values())
# #p= [0.9,0.8,0.7]
# p=1.0 # Do comprehensive search to get canonical values
# ecoli_res = gt.motif_significance(ecoli, k=3, n_shuffles=500, p=p, motif_list=m13, full_output=True)
# yeast_res = gt.motif_significance(yeast, k=3, n_shuffles=500, p=p, motif_list=m13, full_output=True)
# celegans_res = gt.motif_significance(celegans, k=3, n_shuffles=500, p=p, motif_list=m13, full_output=True)
# tadpole_res = gt.motif_significance(tadpole, k=3, n_shuffles=500, p=p, motif_list=m13, full_output=True)
# ecoli_znorm = np.linalg.norm(ecoli_res[1]) # 18.193266239490832
# ecoli_sp = ecoli_res[1] / ecoli_znorm # [-0.613, -0.377, -0.313, 0.620, -0.010, -0.015, -0.001, -0.001, -0.000, 0.000, 0.000, 0.000, 0.000]
# yeast_znorm = np.linalg.norm(yeast_res[1]) # 24.983090564010947
# yeast_sp = yeast_res[1] / yeast_znorm # [-0.561, -0.453, -0.418, 0.537, -0.008, 0.121, -0.001, 0.040, -0.000, 0.000, 0.040, 0.000, 0.000]
# celegans_znorm = np.linalg.norm(celegans_res[1]) #55.00486133710431
# celegans_sp = celegans_res[1] / celegans_znorm #[-0.341, -0.364, -0.230, 0.001, 0.145, 0.100, -0.151, 0.552, 0.367, 0.179, 0.079, 0.316, 0.259]
# tadpole_znorm = np.linalg.norm(tadpole_res[1]) # 56.78274753598695
# tadpole_sp = tadpole_res[1] / tadpole_znorm # [-0.323, -0.289, -0.242, 0.040, 0.038, 0.068, -0.092, 0.378, 0.313, 0.062, 0.154, 0.350, 0.589]

# # Canonicalvalues (gt order)
# ecoli_sp = np.array([-0.613, -0.377, -0.313, 0.620, -0.010, -0.015, -0.001, -0.001, -0.000, 0.000, 0.000, 0.000, 0.000])
# ecoli_znorm = 18.193266239490832
# yeast_sp = np.array([-0.561, -0.453, -0.418, 0.537, -0.008, 0.121, -0.001, 0.040, -0.000, 0.000, 0.040, 0.000, 0.000])
# yeast_znorm = 24.983090564010947
# celegans_sp = np.array([-0.341, -0.364, -0.230, 0.001, 0.145, 0.100, -0.151, 0.552, 0.367, 0.179, 0.079, 0.316, 0.259])
# celegans_znorm = 55.00486133710431
# tadpole_sp = np.array([-0.323, -0.289, -0.242, 0.040, 0.038, 0.068, -0.092, 0.378, 0.313, 0.062, 0.154, 0.350, 0.589])
# tadpole_znorm = 56.78274753598695
# eom_sortix = [motif_sort('eom').index(k) for k in motif_sort('milo')]
# eom_metabolic = np.array([-0.143,-0.453,-0.22,-0.237,0.287,0.017,-0.147,-0.303,0.05,0.393,-0.077,0.153,0.267])
# triad_sortix = [motif_sort('gt').index(k) for k in motif_sort('milo')]
# fig,ax = plt.subplots(figsize=(8,5))
# x = list(range(1,14))
# ax.plot(x, ecoli_sp[triad_sortix],'ro-', label=f'E.Coli transcription\n|Z|={ecoli_znorm:.2f}')
# ax.plot(x, yeast_sp[triad_sortix],'k^--', label=f'Yeast transcription\n|Z|={yeast_znorm:.2f}')
# ax.plot(x, celegans_sp[triad_sortix],'bx-',label=f'C.Elegans neural\n|Z|={celegans_znorm:.2f}')
# ax.plot(x, tadpole_sp[triad_sortix],'g*--',label=f'C. Intestinalis neural\n|Z|={tadpole_znorm:.2f}')
# #ax.plot(x, eom_metabolic[eom_sortix],'m+:', label='Average of 43 metabolic networks')
# ax.grid(True)
# ax.legend()
# ax.set_title('Triad Significance Profile')
# ax.set_ylabel('Normalised Z-Score')
# ax.set_xticks(x)
# ax.set_xlabel('Triad ID')
# #xticklabels_13motifs(ax)
# #plt.switch_backend('cairo')
# fig.savefig('NaturalSignificanceProfile2.pdf')
