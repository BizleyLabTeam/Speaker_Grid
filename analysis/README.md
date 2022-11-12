## Analysis

### Step 1: Spike Extraction and Stimulus Alginment to MCS time

Data is first exported from MCS format to HDF5 using MCS Data Manager.

In matlab, we then extract spike times and obtain the time of stimulus presentation using **[get_PSTHs_for_Recording.m](./Neural/get_PSTHs_for_Recording.m)**. MCS time is the time of stimulus presentation measured using the MCS clock (which is more accurate than the estimated presentation time made in Matlab). This will create a matlab file with spike times and other metadata associated with signal statistics (not of primary interest here). Also generated are summary peri-stimulus time histograms and rasters, that allow a quick assessment of recording quality and units of interest. 

We then run **[export_spike_data_for_py.m](./export_spike_data_for_py.m)** to save the spike times stored in the .mat file into something more python friendly (text files [[example](../data/F1810_Ursula_Squid/2021-05-27_Squid_18-38/spike_times/2021-05-27T18-37-54_SpikeTimes_2_C01.txt)]) and copy the data into the github repo for analysis off site. 


### Step 2: LED tracking and synchronization

Use **[trackLEDs.m](./Video/trackLEDs.m)** to estimate the positions of red and blue LEDs in each video frame. 

**[video_analysis.m](./video_analysis.m)**

Takes as input csv files (**{Datetime}_StimData_MCSAligned.csv** [[example]](../data/F1901_Crumble_Squid/2021-05-31_Squid_17-09/2021-05-31T17-09-46_StimData_MCSAligned.csv)), which were previously named using the format {Datetime}_StimulusData.csv (but this now outdated).

The script uses the synchronization LED within the image to estimate the temporal alignment between video frames and stimulus pulses. Each stimulus generates an LED pulse, which may be detected by the camera or may be missed. The length of the pulse is designed to increase chances of a frame occuring, although this cannot be made too long or the pulse might cover multiple frames (and the longer the pulse is, the greater the uncertainty about the point in the pulse represented by the frame). 

For every session, we return a new table (**{Datetime}_StimData_MCSVidAlign.csv** [[example]](../data/F1810_Ursula_Squid/2021-05-27_Squid_18-38/2021-05-27T18-38-30_StimData_MCSVidAlign.csv)) and graph showing the loss function of the timelag parameter, for which we expect a global minima.



### Step 3: Estimating head pose and relative speaker position

**[combine_head_n_speaker_locations.py](./combine_head_n_speaker_locations.py)**

For each session, this takes in a csv table (**{Datetime}_StimData_MCSVidAlign.csv** [[example]](../data/F1810_Ursula_Squid/2021-05-27_Squid_18-38/2021-05-27T18-38-30_StimData_MCSVidAlign.csv)) and generates a new table (**{Datetime}_Stim_LED_MCS.csv** [[example]](../data/F1810_Ursula_Squid/2021-05-27_Squid_18-38/2021-05-27T18-38-30_Stim_LED_MCS.csv)) that contains additional columns for LED positions, estimated head position and angle between head and speaker. 


### Step 4: Count spikes after each

**[combine_stim_n_spikes.py](./combine_stim_n_spikes.py)**

Takes in the data generated in step 3 (files with name like **{Datetime}_Stim_LED_MCS.csv** [[example]](../data/F1810_Ursula_Squid/2021-05-27_Squid_18-38/2021-05-27T18-38-30_Stim_LED_MCS.csv)) and generates csv tables with spike counts for every channel of recorded data with (**{Datetime}_StimSpikeCounts.csv** [[example]](../data/F1810_Ursula_Squid/2021-05-27_Squid_18-38/2021-05-27T18-38-30_StimSpikeCounts.csv)). Recorded data comes from both left and right headstages, as denoted by the prefix A or B, and channels are numbered according to the MCS system.


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