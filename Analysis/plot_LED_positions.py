"""
Draw LED positions for session

"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
led_path = "/home/stephen/Github/Speaker_Grid/data"
led_file = "2021-05-27_SquidVid_17_30_23.csv"
df = pd.read_csv( os.path.join( led_path, led_file))

# Remove results outside region of interest
df = df[(df['blue_x'] > 150) & (df['blue_x'] < 500)]
df = df[(df['red_x'] > 150) & (df['red_x'] < 500)]

# Compute dwell time
bin_width = 40
half_width = bin_width / 2

x_range = range(150, 540, bin_width)
y_range = range(0, 520, bin_width)

dwell_time = np.zeros((len(y_range), len(x_range)), dtype=int)

for xidx, xbin in enumerate(x_range):

    x_df = df[(df['blue_x'] >= xbin) & (df['blue_x'] < (xbin+bin_width))]

    for yidx, ybin in enumerate(y_range):

        y_df = x_df[(df['blue_y'] >= ybin) & (x_df['blue_y'] < (ybin+bin_width))]

        dwell_time[yidx, xidx] = y_df.shape[0]


# Plot data
fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].scatter(df['red_x'], df['red_y'],s=1, c='#FF0000')
axs[0].scatter(df['blue_x'], df['blue_y'],s=1, c='#00009F')

axs[0].set_xlim(0, 640)
axs[0].set_ylim(0, 480)


axs[1].imshow(np.log(dwell_time))


plt.show()



print(dwell_time.head(n=3))



