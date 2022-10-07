# Speaker_Grid

Data and code associated with pilot testing in the speaker grid - a large environment for sensory neurophysiology experiments.. In the current repo, we only used a small portion of the grid covering 107.5 x 215 cm.

## Get Started

### **data**
Data organized by subject and test session. Data types include:
 
| Type         | Format | Notes                                 | 
| ------------ | ------ | ------------------------------------- |
| LEDs | .csv | positions of LEDs within videos |
| spike times | .txt | times in seconds of spikes for each recording channel (separate files) |
| StimulusData | .csv |  metadata about the time and location of click sounds presented as test stimuli |
| psths | .csv / image | rudimentary plots of spiking activity after sound presentation (useful for checking synchronization of devices) |


### analysis

Scripts to analyse data and plot results - see [documentation](./Analysis/README.md) for more details.

---

### app / assets
Dashboard components (ignore)

### **calibrations**
Images and scripts used to map speaker positions to locations within videos, as well as correct images for fisheye distortion by camera.


### **GoFerret Squid**
Data acquisition software written in Matlab.

### **metadata**
Layout of the grid

