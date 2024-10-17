import numpy as np



def interpolate_states(data_dict):
    """
        I have  a dictionary of lists, data_dict where the key is car_id, representing a car id,
        and the value being a list of its states during the entirety of a predefined time span.
        Each car state is a np array of 10 elements representing the x, y etc of the state.
        Now the problem is, over the entire time span, at some seconds, may be the beginning or
        end or the middle, the state may not have been captured. In this case, I have put a
        placeholder element of [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Please write code that is able to
        interpolate the state given the states before, after or both.
    """
    for car_id, states in data_dict.items():
        states = np.array(states)
        num_states = len(states)
        dim_state = states.shape[1]

        # Identify the indices with missing data
        missing_indices = [i for i, state in enumerate(states) if np.all(state == 0)]

        for idx in missing_indices:
            # Find the indices of the nearest non-missing states before and after the missing state
            prev_idx = next((i for i in range(idx - 1, -1, -1) if i not in missing_indices), None)
            next_idx = next((i for i in range(idx + 1, num_states) if i not in missing_indices), None)

            if prev_idx is not None and next_idx is not None:
                # Interpolate linearly between the states before and after the missing state
                fraction = (idx - prev_idx) / (next_idx - prev_idx)
                states[idx] = (1 - fraction) * states[prev_idx] + fraction * states[next_idx]
                states[idx][-1] = 1
            elif prev_idx is not None:
                # Only a previous state is available, use that
                states[idx] = states[prev_idx]
                states[idx][-1] = 0
            elif next_idx is not None:
                # Only a next state is available, use that
                states[idx] = states[next_idx]
                states[idx][-1] = 0
            # If no adjacent states are available, the missing state remains as is

        # Update the dictionary with interpolated states
        data_dict[car_id] = list(states)

    return data_dict

def interpolate_speed(bounding_boxes, time_interval=0.1):
    """
    Interpolates the speed of 3D bounding boxes.

    Parameters
    ----------
    bounding_boxes : numpy.ndarray
        A numpy array of bounding boxes with shape (N, 7), where N is the number of bounding boxes.
        Each bounding box is represented as (l, w, h, z, y, x, yaw).
    time_interval : float, optional
        The time interval between each bounding box in seconds. Default is 0.1 seconds.

    Returns
    -------
    speeds : numpy.ndarray
        A numpy array of shape (N-1, 2) representing the speed in y and x directions. (v_y, v_x)
        Speeds are calculated as the difference in position divided by the time interval.
    """
    # Calculate the difference in x and y positions
    delta_pos = np.abs(np.diff(bounding_boxes[:, 4:6], axis=0))

    # Divide the change in position by the time interval to get the speed
    speeds = delta_pos / time_interval
    if len(speeds) == 0:
        speeds = np.zeros((1,2))
    else:
        speeds = np.append(speeds, [speeds[-1]], axis=0)
    return speeds