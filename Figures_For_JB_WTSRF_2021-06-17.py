'''


Figures for WT SRF application

'''
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def get_all_head_positions_at_stim():
    """
    Replicates the scatter plot showing head position at the time 
    of each pulse stimulus shown in the dashboard, but concatenates
    across all available data.
    
    Parameters:
    ----------
    a : var_type
        Description
    
    Returns:
    --------
    a : var_type
        Description
    """

    # Load background image
    img = mpimg.imread('calibrations/2021-05-31_SpeakerLocations_Corrected_Gray.tif')

    # Load scatter data
    root_dir = Path('data')
    df = []

    for file in root_dir.rglob('*StimSpikeCounts.csv'):
        df.append(pd.read_csv( str(file)))        

    df = pd.concat(df)
    
    # Plot scatter
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])

    ax.imshow(img)
    ax.scatter(df['blue_x'], df['blue_y'], s=1, c=df['Speaker'], cmap=plt.cm.gist_ncar)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.show()



def main():
    get_all_head_positions_at_stim()


if __name__ == '__main__':
    main()