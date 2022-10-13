""" Estimate the scale factor between pixels and centimeters """

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

fp = 'metadata/Speaker_grid_layout_2021_05_21_filtered.csv'

df = pd.read_csv(fp)

# Create distance matrix between every sound source observed
dist_cm, dist_px = [], []
chans = df['matlab_chan'].unique()

for i in chans:
    for j in chans:
        if i > j:

            # Get data from table as dictionaries
            row_i = df.loc[df[df['matlab_chan'] == i].index].to_dict('records')[0]
            row_j = df.loc[df[df['matlab_chan'] == j].index].to_dict('records')[0]

            cm_x = row_i['Grid_x_cm'] - row_j['Grid_x_cm']
            cm_y = row_i['Grid_y_cm'] - row_j['Grid_y_cm']

            dist_cm.append( (cm_x**2 + cm_y**2) ** 0.5)

            px_x = row_i['2021_05_31_PixLoc_Corrected_x'] - row_j['2021_05_31_PixLoc_Corrected_x']
            px_y = row_i['2021_05_31_PixLoc_Corrected_y'] - row_j['2021_05_31_PixLoc_Corrected_y']

            dist_px.append( (px_x**2 + px_y**2) ** 0.5)


# Perform regression
# slope, intercept = statistics.linear_regression(dist_cm, dist_px)
x = sm.add_constant(dist_cm)
model = sm.OLS(dist_px, x).fit()

print(model.summary())

# Print just for sanity
plt.plot(dist_cm, dist_px,'.')
plt.show()