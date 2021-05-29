function est_error = get_align_error( pulse_times, stim_data, offset)

delta = bsxfun(@minus, pulse_times + offset, stim_data.MCS_Time');
delta = abs(delta);

min_delta = min(delta, [], 2);

est_error = sum(min_delta);