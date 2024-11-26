#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.motifs import motif_sort, target_tsp

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJ_DIR,'results')



def csv_res_plot(res_df, target=None, 
                 show_err: bool = True, 
                 avg_window: int = 0,
                 triad_sort: str = 'milo'):
    fig, axs = plt.subplots(nrows=2, layout='constrained')
    err_names = {'mae': "Mean Absolute Error", 'rmse': "RMSE", 'canberra': "Canberra Distance"}
    err_type = 'MAE' #default
    #cum_avg_fitness =plt.plot(np.cumsum(results['fitness']) / np.array(range(1,len(results['fitness'])+1)))
    if show_err:
        axs[0].plot(res_df['MAE'], ':r', label='Error') 
    if avg_window==0:
        axs[0].plot(res_df['MAE'].expanding().mean(), '--k', label='Average Error') # rolling(2*cfg['evol'].getint('popsize')**2)
    elif avg_window<0:
        # don't plot average
        print('Not plotting rolling average')
    else:
        axs[0].plot(res_df['MAE'].rolling(avg_window).mean(), '--k', label='Average Error') 
    axs[0].plot(res_df['MAE'].expanding().min(), '-b',label='Min. Error')
    axs[0].legend()
    axs[0].set_title('Error over the course of an evolutionary run')
    axs[0].set_ylabel(err_type)
    axs[0].set_xlabel('Trials')
    axs[0].grid(True)
    triad_sortix = [motif_sort('gt').index(k) for k in motif_sort(triad_sort)]
    best_mae = np.min(res_df['MAE'])
    best_idx = np.argmin(res_df['MAE'])
    # pull out zscores
    best_zs = res_df.iloc[best_idx,1:14].to_numpy()
    z_norm = np.linalg.norm(best_zs)
    best_sig_prof = best_zs / z_norm
    x = range(1,14)
    if target is not None:
        target_sig_prof = target_tsp[target]
        axs[1].plot(x,target_sig_prof[triad_sortix], 'o-b', label=f'Target\n({target})')
    axs[1].plot(x,best_sig_prof[triad_sortix], '*--r', label=f'Best (Err: {best_mae:.03f})')
    axs[1].set_xticks(x)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Target and Evolved Triad Significance Profile')
    axs[1].set_ylabel('Normalised Triad Z-Score')
    axs[1].set_xlabel('Triad ID')
    axs[1].grid(True)
    return fig

#%%
fn = 'results_20241125_163856' 
fn = 'results_20241126_085703'
res = pd.read_csv(os.path.join(RESULTS_DIR,fn+'.csv'), 
                  delimiter='\t', header=0,index_col=None,
                  names=['MAE',*[f'Z{i}' for i in range(13)],'Model','Graph','Skipped'])

# %%
csv_res_plot(res, target='ecoli', show_err=False, avg_window=-1)
# %%
