## Analysis

### Step 1: Spike Extraction and Stimulus Alginment to MCS time

Data is first exported from MCS format to HDF5 using MCS Data Manager.

In matlab, we then extract spike times and obtain the time of stimulus presentation using **get_PSTHs_for_Recording.m**. MCS time is the time of stimulus presentation measured using the MCS clock (which is more accurate than the estimated presentation time made in Matlab). This will create a matlab file with spike times and other metadata associated with signal statistics (not of primary interest here). Also generated are summary peri-stimulus time histograms and rasters, that allow a quick assessment of recording quality and units of interest. 

We then run **export_spike_data_for_py.m** to save the spike times stored in the .mat file into something more python friendly (text files) and copy the data into the github repo for analysis off site. 


### Step 2: LED tracking and synchronization

Use **trackLEDs.m** to estimate the positions of red and blue LEDs in each video frame

We then use the synchronization LED within the image to estimate the temporal alignment between video frames and stimulus pulses (**video_analysis.m**). Each stimulus generates an LED pulse, which may be detected by the camera or may be missed. The length of the pulse is designed to increase chances of a frame occuring, although this cannot be made too long or the pulse might cover multiple frames (and the longer the pulse is, the greater the uncertainty about the point in the pulse represented by the frame). The function returns a graph showing the loss function of the timelag parameter, for which we expect a global minima, as well as the distribution of time differences between pulse and 




### Step 3: Estimating head pose and relative speaker position

**combine_head_n_speaker_locations.py**


#### *Notes*

Spike extraction here does not use any cleaning and so the data can look a little noisy. Future updates may add cleaning.

With this camera, there are also steps we can take to make the tracking better (e.g. max saturation, play with brightness, contrast and gamma). We could also introduce some minor interpolation to pick up small runs in frames that the LEDs go out of view.

Due to the size of files, videos and neural recording are not included in the repo.



## Metadata

Left hemisphere = {
	"Headstage" : 'H115',
	"Rotation": "Normal",
	"File_stub" : "McsRecording"
}

Right hemisphere = {
	"Headstage" : 'H109',
	"Rotation": "Rotated 180",
	"File_stub" : "McsRecording_2"
}


Area of 107.5 x 215 cm 