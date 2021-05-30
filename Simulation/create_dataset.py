"""

Create a simulated dataset to explore the possible outcomes of
experiments with the speaker grid


Created on 29 May 2021 by Stephen Town (s.town@ucl.ac.uk)

"""


import pandas as pd

my_list = []

for speaker in range(0, 48):
    for head_stim_angle in range(-150, 210, 30):
        my_dict = {
        'speaker': speaker,
        'head_stim_angle': head_stim_angle,
        'response': head_stim_angle
        }

        my_list.append(my_dict)

df = pd.DataFrame(my_list)

df.to_csv('sim_dataset.csv', index=False)
