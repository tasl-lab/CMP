import os
import pickle

scenes = os.listdir()
scenes = sorted([x for x in os.listdir() if x.endswith("pickle")])

for scene in scenes[1:2]:
    if not scene.endswith("pickle"):
        continue
    f = open(scene, 'rb') # testoutput_CAV_data_2022-03-21-09-35-07_7-1-traj.pickle
    data = pickle.load(f)
    scenario_name = ('-').join(scene.split('-')[:-2])
    cav_id = scene.split('-')[-2]
    # if int(cav_id) == 1:
    #     continue

    timestamps = len(data['timestamps'])
    timestamp_key = data['timestamps']
    data = data['data']

    for states in data[1]:
        print(states)

    print(scene)
    print(data.keys())