function get_pixel_loc_for_speakers

% Load calibration image
im = imread('2021-05-31_SpeakerLocations_Corrected.tif');

figure; imshow(im)
roi = getrect(gcf);

roi_centre(1) = roi(1) + (roi(3)/2);
roi_centre(2) = roi(2) + (roi(4)/2);

hold on
plot(roi_centre(1),roi_centre(2), 'ok','markerfacecolor','k')
fprintf('%f\n', roi_centre)