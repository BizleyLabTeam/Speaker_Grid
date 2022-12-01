

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

    x_kernel = res['x_kernel'].tolist()
    y_kernel = res['y_kernel'].tolist()
    ypCI_kernel = res['ypCI_kernel'].tolist()
    ymCI_kernel = res['ymCI_kernel'].tolist()

    ax.plot(x_kernel, y_kernel, color='r')
    ax.fill_between(x_kernel, ymCI_kernel, ypCI_kernel, color='r', alpha=0.3)

    if ylim:
        ax.set_ylim(ylim)
        ax.set_yticks(ylim)


def polar_plot(file_path, unit, var, ax, ylim:Optional=None):

    # Load file 
    file_name = f"{unit}_{var}.npz"
    res = np.load(file_path / file_name, allow_pickle=True)

    x_model = res['x_model'].tolist()
    x_raw = res['x_raw'].tolist()

    y_model = res['y_model'].tolist()
    y_raw = res['y_raw'].tolist()
    
    ax.plot(x_model, y_model, label='model')
    # ax.scatter(raw_theta, raw_rho, s=3, alpha=0.8, color='orange',label='raw')
    ax.bar(x_raw, y_raw, width=np.pi/12, bottom=0.0, color='k', alpha=0.65)

    if ylim:
        ax.set_ylim(ylim)
        ax.set_yticks(ylim)



def remove_xticklabels(ax_list):

    for ax in ax_list:
        ax.set_xticklabels('')



def main():

    data_path = Path('/home/stephen/Github/PGAM/tests/output')
    
    fig = plt.figure(figsize=(4, 4))


    gs = fig.add_gridspec(nrows=4, ncols=2, left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.9)

    # Polar plot axes
    ppax = dict(
        B09 = dict(
            HA = add_custom_subplot(fig,gs[0,1]),      
            srf = add_custom_subplot(fig,gs[0,0]),
        ),
        B13 = dict(
            HA = add_custom_subplot(fig,gs[2,1]),
            srf = add_custom_subplot(fig,gs[2,0]) 
        )
    )

    # Model confidence axes
    mdl_axes = dict(
        B09 = dict(
            HA = add_custom_subplot(fig,gs[1,1]),      
            srf = add_custom_subplot(fig,gs[1,0]),
        ),
        B13 = dict(
            HA = add_custom_subplot(fig,gs[3,1]),
            srf = add_custom_subplot(fig,gs[3,0]) 
        )
    )

    # intx_axes['A25_to_A22'].yaxis.tick_right()
    polar_plot(data_path, unit='B09', var='Head_direction', ax=ppax['B09']['HA'])
    polar_plot(data_path, unit='B13', var='Head_direction', ax=ppax['B13']['HA'])

    polar_plot(data_path, unit='B09', var='SRF', ax=ppax['B09']['srf'])
    polar_plot(data_path, unit='B13', var='SRF', ax=ppax['B13']['srf'])

    plot_model_confidence(data_path, unit='B09', var='Head_direction', ax=mdl_axes['B09']['HA'])
    plot_model_confidence(data_path, unit='B13', var='Head_direction', ax=mdl_axes['B13']['HA'])

    plot_model_confidence(data_path, unit='B09', var='SRF', ax=mdl_axes['B09']['srf'])
    plot_model_confidence(data_path, unit='B13', var='SRF', ax=mdl_axes['B13']['srf'])
    

    # remove_xticklabels([
    #     intx_axes['A25_to_A22'],
    #     psth_axes['A25'], 
    #     psth_axes['B13'],
    #     intx_axes['A25_to_B13'],
    #     intx_axes['B13_to_B09']])


    # plt.savefig('tuning.png', dpi=300)
    plt.show()


    


if __name__ == '__main__':
    main()