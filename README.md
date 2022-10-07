# Speaker_Grid

Data and code associated with pilot testing in the speaker grid - a large environment for sensory neurophysiology experiments.. In the current repo, we only used a small portion of the grid covering 107.5 x 215 cm.

## Get Started

### Analysis

### **calibrations**
Images and scripts used to map speaker positions to locations within videos, as well as correct images for fisheye distortion by camera.

### **data**
Data organized by subject and test session. Data types include:
* LEDs: csv files containing positions of LEDs within videos
* spike times: text files for each recording channel
* StimulusData: csv file containing metadata about the time and location of click sounds presented as test stimuli
* psths: csv files and images showing rudimentary plots of spiking activity after sound presentation (useful for checking synchronization of devices)

### **GoFerret Squid**
Data acquisition software written in Matlab.

