""" Tools to analyse the position of an animal """

from typing import Optional, Tuple


def get_dwell_time(df:pd.DataFrame, bin_width:int=40, xrange:Tuple[int,int]=(150, 540),
    yrange:Tuple[int,int]=(0, 520), fps:Optional[float]=None)-> np.ndarray:
    """ 
    Get the number of frames in which blue LEDs dwelt spent in each location in 
    binned space.

    Args:
        df: dataframe with LED positions in pixels as columns ['blue_x','blue_y]
        bin_width: size of spatial bin in pixels
        xrange: upper and lower bounds of image in which to compute dwell time
        yrange: upper and lower bounds of image in which to compute dwell time
        fps: frame rate, if None, code will return dwell times in frames rather than seconds
    
    Returns:
        dwell_time: 2d array of 

     """

    # Preassign results grid
    half_width = bin_width / 2
    xrange = range(xrange[0], xrange[1], bin_width)
    yrange = range(0, 520, bin_width)

    dwell_time = np.zeros((len(yrange), len(xrange)), dtype=int)

    # For each bin
    for xidx, xbin in enumerate(xrange):

        x_df = df[(df['blue_x'] >= xbin) & (df['blue_x'] < (xbin+bin_width))]

        for yidx, ybin in enumerate(yrange):

            y_df = x_df[(df['blue_y'] >= ybin) & (x_df['blue_y'] < (ybin+bin_width))]

            dwell_time[yidx, xidx] = y_df.shape[0]

    # Optional convert from frames to seconds
    if fps is not None:
        dwell_time = dwell_time.astype(float) / fps

    return dwell_time


def main():
    pass


if __name__ == '__main__':
    main()