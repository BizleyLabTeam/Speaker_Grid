

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
def plot_interaction(file_path, ax, ylim:Optional=None):

    res = np.load(file_path, allow_pickle=True)

    x = res['x'].tolist()
    yraw = res['y_raw'].tolist()
    ymdl = res['y_model'].tolist()

    ax.plot( x, yraw, color='#1a2e4b', lw=0.8)
    ax.plot( x, ymdl, color='orange', lw=1)
    
    ax.plot([0, 0], ax.get_ylim(), zorder=0, ls='--', lw=0.8)
    # ax.set_xlim((min(x), max(x)))
    ax.set_xlim((-0.5, 0.5))
    ax.set_xticks([-0.4, 0, 0.4])

    if ylim:
        ax.set_ylim(ylim)
        ax.set_yticks(ylim)


def plot_psth(file_path, ax, ylim:Optional=None):

    res = np.load(file_path, allow_pickle=True)

    x = res['x'].tolist()
    yraw = res['y_raw'].tolist()
    ymdl = res['y_model'].tolist()

    ax.bar( x, yraw, color='#998877', width=0.012, edgecolor='none')
    ax.plot( x, ymdl, color='red', lw=1)
    
    ax.plot([0, 0], ax.get_ylim(), zorder=0, ls='--', lw=0.8)
    ax.set_xlim((-0.15, 0.15))

    ax.set_xticks([-0.1, 0, 0.1])
    
    if ylim:
        ax.set_ylim(ylim)
        ax.set_yticks(ylim)



def remove_xticklabels(ax_list):

    for ax in ax_list:
        ax.set_xticklabels('')



def main():

    data_path = Path('/home/stephen/Github/PGAM/tests/output')
    
    fig = plt.figure(figsize=(2, 2))


    gs = fig.add_gridspec(nrows=3, ncols=3, left=0.1, right=0.9, bottom=0.1, top=0.8, hspace=0.8)

    psth_axes = dict(
        A25 = add_custom_subplot(fig,gs[0,0]),
        A22 = add_custom_subplot(fig,gs[2,0]),      
        B09 = add_custom_subplot(fig,gs[2,2]),
        B13 = add_custom_subplot(fig,gs[0,2]) 
    )

    intx_axes = dict(
        A25_to_A22 = add_custom_subplot(fig,gs[1,0]),      
        A25_to_B13 = add_custom_subplot(fig,gs[0,1]),
        B13_to_B09 = add_custom_subplot(fig,gs[1,2]),
        A22_to_B09 = add_custom_subplot(fig,gs[2,1]) 
    )

    intx_axcolor = '#667799'
    recolor_axes(intx_axes, color=intx_axcolor)
    
    # add_axes_titles(psth_axes)
    # add_axes_titles(intx_axes, color=intx_axcolor)

    # Plot data
    plot_interaction(data_path/'A25_to_A22.npz', intx_axes['A25_to_A22'], ylim=(46,57)) 
    plot_interaction(data_path/'B13_to_B09.npz', intx_axes['B13_to_B09'], ylim=(0,20)) 
    plot_interaction(data_path/'A25_to_B13.npz', intx_axes['A25_to_B13'], ylim=(0,15)) 
    plot_interaction(data_path/'A22_to_B09.npz', intx_axes['A22_to_B09'], ylim=(0,5)) 

    # intx_axes['A25_to_A22'].yaxis.tick_right()

    plot_psth(data_path/'A22_PSTH.npz', psth_axes['A22'], ylim=(40,70)) 
    plot_psth(data_path/'A25_PSTH.npz', psth_axes['A25'], ylim=(0,40)) 
    plot_psth(data_path/'B09_PSTH.npz', psth_axes['B09'], ylim=(0,5)) 
    plot_psth(data_path/'B13_PSTH.npz', psth_axes['B13'], ylim=(0,40))

    remove_xticklabels([
        intx_axes['A25_to_A22'],
        psth_axes['A25'], 
        psth_axes['B13'],
        intx_axes['A25_to_B13'],
        intx_axes['B13_to_B09']])


    plt.savefig('interactions.png', dpi=300)
    plt.show()


    


if __name__ == '__main__':
    main()