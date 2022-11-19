

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

    ax = fig.add_subplot(gs, polar=True)
    
    ax.tick_params(direction='in', length=1.5, pad=2)

    ax.spines['polar'].set_linewidth(0.2)

    # for axis in ['bottom','left']:
    #     ax.spines[axis].set_linewidth(0.8)

    return ax

def add_axes_titles(axs:dict, color:str='k') -> None:

    for tstr, ax in axs.items():
        ax.set_title(tstr, color=color)





def polar_plot(file_path, unit, var, ax, zmax):

    # Load file 
    file_name = f"{unit}_{var}.npz"
    res = np.load(file_path / file_name, allow_pickle=True)

    if var == 'SRF':
        x_firing = res['sound_angle'].tolist()
    elif var == 'Head_direction':
        x_firing = res['head_angle'].tolist()

    if var == 'Head_direction':
        line_color = 'magenta'
        scat_color = 'green'
    else:
        line_color = '#1f77b4'
        scat_color = 'orange'
    
    if unit == 'B09':
        scat_alpha = 1
    else:
        scat_alpha = 0.3

    y_firing_model = res['y_model'].tolist()
    raw_theta = res['raw_theta'].tolist()
    raw_rho = res['raw_rho'].tolist()

    ax.plot(x_firing, y_firing_model, label='model', color=line_color, lw=1)
    ax.scatter(raw_theta, raw_rho, s=0.5, alpha=scat_alpha, color=scat_color,label='raw')

    ax.set_rmax(zmax)
    ax.set_rmin(0)
    ax.set_theta_offset(np.pi/2)
    ax.grid(True, color='#a8a886', lw=0.5)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_facecolor("#eeeeee")



def remove_xticklabels(ax_list):

    for ax in ax_list:
        ax.set_xticklabels('')



def main():

    data_path = Path('/home/stephen/Github/PGAM/tests/output')
    
    fig = plt.figure(figsize=(8, 8))


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


    # intx_axes['A25_to_A22'].yaxis.tick_right()
    polar_plot(data_path, unit='B09', var='Head_direction', ax=ppax['B09']['HA'], zmax=60)
    polar_plot(data_path, unit='B13', var='Head_direction', ax=ppax['B13']['HA'], zmax=250)

    polar_plot(data_path, unit='B09', var='SRF', ax=ppax['B09']['srf'], zmax=60)
    polar_plot(data_path, unit='B13', var='SRF', ax=ppax['B13']['srf'], zmax=250)

    plt.savefig('polar_tuning.png', dpi=300)
    plt.show()


    


if __name__ == '__main__':
    main()