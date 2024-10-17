import os
import pickle
import matplotlib.pyplot as plt

# Get all the scene files
scenes = sorted([x for x in os.listdir() if x.endswith("pickle")])

for scene in scenes:
    if not scene.endswith("pickle"):
        continue

    # Open the pickle file and load the data
    with open(scene, 'rb') as f:
        data = pickle.load(f)

    scenario_name = ('-').join(scene.split('-')[:-2])
    cav_id = scene.split('-')[-2]
    
    # Skip cav_id 1
    # if int(cav_id) == 1:
    #     continue

    # Extract timestamps and trajectory data
    timestamps = len(data['timestamps'])
    timestamp_key = data['timestamps']
    data = data['data']

    # Create a figure for the plot
    plt.figure(figsize=(10, 10))

    # Loop over all objects (obj_id) in the scene
    for obj_id, states in data.items():
        # if obj_id > 1000:
        #     continue
        x_vals = []
        y_vals = []

        # Loop over states for each object
        for state in states:
            x, y, z, l, w, h, theta, valid = state  # unpack the state

            # Only keep valid states
            if valid == 1:
                x_vals.append(x)
                y_vals.append(y)

        # Plot the valid trajectory for this obj_id
        if x_vals and y_vals:  # only plot if we have valid data
            plt.plot(x_vals, y_vals, label=f'Object {obj_id}')
    
    plt.axis('equal')

    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Trajectories for Scenario {scenario_name}, CAV {cav_id}')
    plt.legend()

    # Save the plot to a file instead of displaying it
    plot_filename = f"plot/{scenario_name}_CAV_{cav_id}_trajectory.png"
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory

    print(f"Saved plot for {scene} as {plot_filename}")
