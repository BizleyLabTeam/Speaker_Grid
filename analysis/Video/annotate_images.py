""" Add labels showing LED position, head position and direction for sample frames 

Notes:
    Video files aren't included with the repository - contact Stephen if you need them (or they're on G:/UCL_Behaving on the Squid)

"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

# sys.path.insert(0, str(Path.cwd()))


# Settings
ferret = 'F1901_Crumble'
session = '2021-05-31_Squid_17-09'


def load_tracking_file(file_path) -> pd.DataFrame:
    """ Load tracking data from csv file with specific suffix in file name """
    
    files = file_path.glob('*Stim_LED_MCS.csv')
    return pd.read_csv(next(files))


def find_video_file(file_path:Path, session:str):
    """ Find video file based on prefix in file name inherited from session datetime """

    # Reformat session name for video filename style
    stub = session.replace('Squid','SquidVid')

    files = file_path.glob(stub + "*")
    return cv2.VideoCapture(str(next(files)))

def mark_LEDs(frame, x, y, color):

    return cv2.drawMarker(
        frame, 
        (int(x),int(y)), 
        color=color, 
        markerType=cv2.MARKER_CROSS, 
        thickness=1)



def main():

    # Load tracking data
    file_path = Path.cwd() / 'data' / f"{ferret}_Squid" / session
    df = load_tracking_file(file_path)

    # Get video file
    video_path = Path.cwd() / 'videos'
    cap = find_video_file(video_path, session)

    # Create directory for saving images
    new_dir = file_path / 'head_angle_snaps'
    new_dir.mkdir(exist_ok=True)

    # For every 30° around the circle, plot an example image from each
    for deg in range(-180, 180, 30):

        # Find frame with head angle closest to our target
        rad = (deg / 180) * np.pi
        idx = np.argmin(np.abs(df.head_angle - rad))
        trial_info = df.loc[idx]

        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(trial_info.closest_frame))
        ret, frame = cap.read()

        # Label LEDs
        mark_LEDs(frame, trial_info.red_x, trial_info.red_y, (0, 0, 255))
        mark_LEDs(frame, trial_info.blue_x, trial_info.blue_y, (255, 0, 5))

        cv2.line(
            frame, 
            (int(trial_info.blue_x), int(trial_info.blue_y)),
            (int(trial_info.red_x), int(trial_info.red_y)),
            color = (255, 255, 255),
            thickness = 1
        )

        # Save 
        save_file = new_dir / f"Check_{deg}.jpg"
        cv2.imwrite(str(save_file), frame)

    


if __name__ == '__main__':
    main()