#%%
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import graph_tool.all as gt
from src.motifs import motif_sort

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJ_DIR,'results')

def plan_tuple_plot_signifigance_profile(zscores: np.ndarray, ax, 
                                         sort_type: int = 0, 
                                         horizontal: bool = False):
    """
    Expects zscores to be an trials*motifs array
    """
    znorms = np.linalg.norm(zscores, axis=1, keepdims=True)
    sp = zscores / znorms
    #sp = np.array([significance_profile(r[0]) for r in results])
    x = np.array(range(1,14))
    if sort_type==1:
        # just sort rows by znorm and cols by mean
        rowsort = np.argsort(np.squeeze(znorms)) # better?
        colsort = np.argsort(np.mean(sp, axis=0))
        sorted = sp[rowsort,:][:,colsort]
        xlabels = x[colsort]
    elif sort_type==2:
        #sort first by biggest sp, then by znorm (lexsort takes primary key last, weirdly)
        rowsort = np.lexsort([np.argsort(np.squeeze(znorms)), np.argmax(sp, axis=1)])
        sorted = sp[rowsort,:]
        xlabels = x
    elif sort_type==3:
        # rows by biggest sp1, then biggest sp2 etc.
        rowsort = np.lexsort(np.argsort(sp).T.tolist())
        sorted = sp[rowsort,:]
        xlabels = x
    if horizontal:
        c = ax.imshow(sorted.T, cmap='bwr',vmax=1.0, vmin=-1.0) # to get Susan's "plan tuple plot"
        ax.set_yticks(x-1, labels=xlabels)
    else:
        c = ax.imshow(sorted, cmap='bwr',vmax=1.0, vmin=-1.0) # to get Susan's "plan tuple plot"
        ax.set_xticks(x-1, labels=xlabels)
    cax = fig.colorbar(c, ax=ax,shrink=0.3)
    cax.set_label('Normalised Z-Score')

#%%

#fn = 'results_20241120_181917.p'
#fn = 'results_20241122_154827.p'
#fn = 'results_20241122_173057.p'
fn = 'results_20241124_225236.p'
with open(os.path.join(RESULTS_DIR,fn), 'rb') as handle:
    results = pickle.load(handle)

#%%
zscores = np.array([r[0] for r in results])
triad_sortix = [motif_sort('gt').index(k) for k in motif_sort('milo')]
zscores = zscores[:, triad_sortix]
fig, ax = plt.subplots(figsize=(12,6))
plan_tuple_plot_signifigance_profile(zscores, ax, sort_type=3, horizontal=True)
ax.set_xticks(list(range(0,101,10)))
ax.set_yticks(list(range(0,14,2)))
ax.set_title('Triad Significance Profile')
ax.set_ylabel('Motif Number')
ax.set_xlabel('Grown Graphs (sorted by Significance Profile)')
fig.savefig(os.path.join(RESULTS_DIR,fn[:-2]+'.pdf'))

# %%
