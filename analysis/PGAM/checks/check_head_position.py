""" Plot head positions to see what range of pixel values we should specify knots for"""


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import seaborn as sns

sys.path.insert(0, str(Path.cwd()))
from lib import utils

fontsize=6
plt.rc('font', size=fontsize) #controls default text size
plt.rc('axes', titlesize=fontsize+1) #fontsize of the title
plt.rc('axes', labelsize=fontsize) #fontsize of the x and y labels
plt.rc('xtick', labelsize=fontsize) #fontsize of the x tick labels
plt.rc('ytick', labelsize=fontsize) #fontsize of the y tick labels
plt.rc('legend', fontsize=fontsize) #fontsize of the legend


def plot_positions(x, y):

    # Create canvas
    fig, ax = plt.subplots(nrows=1, ncols=1)
        
    ax.plot(psth, lw=0.7)
    ax.set_title(neu_name, pad=0)

    # plt.tight_layout()
    plt.show()
    
    
def load_LED_data(data_path:Path) -> pd.DataFrame:
    """ Load tracking data from DeepLabCut """
    
    LEDs = []

    for session in data_path.glob('*Squid*'):

        LED_path = session / 'LEDs'
        LED_file = next( LED_path.glob('*correctedDLC*.csv'))

        LEDs.append( 
            utils.read_deeplabcut_csv(
                file_path=LED_path,
                file_name=LED_file.name
            )
        )
        
    return pd.concat(LEDs)


def get_confidence_values(df:pd.DataFrame) -> pd.DataFrame:
    """ Estimate how confident we are about the LED position """
    return (
        df
        .assign(blue_LEDlikelihood = lambda x: (x.blue_xa / x.blue_xc) + (x.blue_ya / x.blue_yc))       # This isn't a likelihood in the statistical sense
        .assign(red_LEDlikelihood = lambda x: (x.red_xa / x.red_xc) + (x.red_ya / x.red_yc))
    )



def filter_for_low_likelihoods(df:pd.DataFrame, threshold:float) -> pd.DataFrame:
    """ Remove tracking values for which estimation was uncertain
    
    Note that estimation methods for Matlab and DeepLabCut are different and 
    thus require different values to assess.
    
     """

    # Get indices of trials to mask, independently for red and blue LEDs
    blue_idx = df['blue_LEDlikelihood'] < threshold
    red_idx = df['red_LEDlikelihood'] < threshold

    for var in ['_LEDx','_LEDy']:
        
        df['blue'+var].mask(blue_idx, inplace=True)
        df['red'+var].mask(red_idx, inplace=True)

    return df


def process_headtrack(df:pd.DataFrame)-> pd.DataFrame:
    """ 
    
    To do:
        Implement optional filter
     """

    # Remove values with low likelihoods
    likelihood_threshold = 0.6
    idx = df[df['blue_LEDlikelihood'] < likelihood_threshold].index

    df.loc[idx, 'blue_LEDx'] = np.nan
    df.loc[idx, 'blue_LEDy'] = np.nan

    # Interpolate missing values
    df['blue_LEDx'].interpolate(method='nearest', inplace=True)
    df['blue_LEDy'].interpolate(method='nearest', inplace=True)

    return df


def main():

    # Settings
    ferret = 'F1901_Crumble'

    # Load data for each session
    data_path = Path.cwd() / 'data' / f"{ferret}_Squid"
    LEDs = load_LED_data(data_path)

    # LEDs = get_confidence_values(LEDs)
    # LEDs = filter_for_low_likelihoods(LEDs, 0.6)
    LEDs = process_headtrack(LEDs)
    

    fig = plt.figure(figsize=(4.72, 1.97))
    
    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(0.225), Size.Fixed(0.8)]
    v = [Size.Fixed(0.4), Size.Fixed(1.2)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(),
                    axes_locator=divider.new_locator(nx=1, ny=1))

    ax.set_facecolor("#d5d5aa")

    for i in ['left','top','bottom','right']:
        ax.spines[i].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    # Show spike counts around event times (predict events should drive change in counts)
    sns.scatterplot(
        data = LEDs,
        x='blue_LEDx',
        y='blue_LEDy', 
        hue = 'blue_LEDlikelihood',
        alpha=0.4,
        edgecolor='none',
        palette='Purples',
        marker = '.',
        s=1,
        ax=ax,
        legend=False
        )

    ax.set_xlim((160, 440))
    ax.set_ylim((0, 480))

    plt.savefig('head_positions_final.png', dpi=300)
    plt.show()




if __name__ == '__main__':
    main()
