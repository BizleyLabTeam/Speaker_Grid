

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

fontsize=6
plt.rc('font', size=fontsize) #controls default text size
plt.rc('axes', titlesize=fontsize+1) #fontsize of the title
plt.rc('axes', labelsize=fontsize) #fontsize of the x and y labels
plt.rc('xtick', labelsize=5) #fontsize of the x tick labels
plt.rc('ytick', labelsize=5) #fontsize of the y tick labels
plt.rc('legend', fontsize=fontsize) #fontsize of the legend



def add_custom_subplot(fig, gs):

    ax = fig.add_subplot(gs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(direction='in', length=1.5, pad=2)

    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(0.8)

    return ax

def add_axes_titles(axs:dict, color:str='k') -> None:

    for tstr, ax in axs.items():
        ax.set_title(tstr, color=color)


def recolor_axes(axs, color):
    # Use a different color for interaction axes
    for ax in axs.values():
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_color(color)

        ax.tick_params(axis='x', colors=color)
        ax.tick_params(axis='y', colors=color)


#############################################################################
def plot_model_confidence(file_path, unit, var, ax, ylim:Optional=None):

    # Load file 
    file_name = f"{unit}_{var}.npz"
    res = np.load(file_path / file_name, allow_pickle=True)

    if var == 'Head_direction':
        color = 'magenta'
    else:
        color = '#1f77b4'

    x_kernel = res['x_kernel'].tolist()
    y_kernel = res['y_kernel'].tolist()
    ypCI_kernel = res['ypCI_kernel'].tolist()
    ymCI_kernel = res['ymCI_kernel'].tolist()

    ax.plot([-np.pi, np.pi], [0,0], color='k',ls='--',lw=0.8)
    ax.plot(x_kernel, y_kernel, color=color,lw=0.8)
    ax.fill_between(x_kernel, ymCI_kernel, ypCI_kernel, color=color, alpha=0.3)

    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(['-180', '0', '180'])
    ax.set_xlim((-np.pi, np.pi))

    if ylim:
        ax.set_ylim(ylim)
        ax.set_yticks(ylim)





def remove_xticklabels(ax_list):

    for ax in ax_list:
        ax.set_xticklabels('')

def main():

    data_path = Path('/home/stephen/Github/PGAM/tests/output')
    
    fig, axs = plt.subplots(2,2, sharey=True, sharex=True, **{'figsize':(1.45, 0.8)})

    # Model confidence axes
    mdl_axes = dict(
        B09 = dict(
            HA = add_custom_subplot(fig, axs[0,1]),      
            srf = add_custom_subplot(fig, axs[0,0]),
        ),
        B13 = dict(
            HA = add_custom_subplot(fig, axs[1,1]),
            srf = add_custom_subplot(fig, axs[1,0]) 
        )
    )

    plot_model_confidence(data_path, unit='B09', var='Head_direction', ax=mdl_axes['B09']['HA'])
    plot_model_confidence(data_path, unit='B13', var='Head_direction', ax=mdl_axes['B13']['HA'])

    plot_model_confidence(data_path, unit='B09', var='SRF', ax=mdl_axes['B09']['srf'])
    plot_model_confidence(data_path, unit='B13', var='SRF', ax=mdl_axes['B13']['srf'])
    
    plt.tight_layout()
    plt.savefig('tuning_model.png', dpi=300)
    plt.show()


    


if __name__ == '__main__':
    main()