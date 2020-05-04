import pandas as pd
import numpy as np
import argparse
import datetime

def group_point_per_sec(timestamps):
    groups = []
    curr_second = 0
    curr_count = 0
    for timestamp in timestamps:
        d = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
        if d.second != curr_second:
            groups.append(curr_count)
            curr_second = d.second
            curr_count = 0
        curr_count += 1
    return groups


parser = argparse.ArgumentParser(
    description='Extraxt info from mocked data')
parser.add_argument('sensor_data_path',  type=str,
                    help='path of mocked data csv')
parser.add_argument('frames_data_path',  type=str,
                    help='path of mocked video')
args = parser.parse_args()

senors_data = pd.read_csv(
    args.sensor_data_path,  names=['timestamp', 'x', 'y', 'z', 'lat', 'lng', 'target'])
frames_index = pd.read_csv(args.frames_data_path, names=[
    'timestamp', 'frame_index'])


print("#"*100)
print("--SENSORS DATA--")
print("- path :", args.sensor_data_path)
print("- data frame csv:")
print(senors_data)
senors_data = senors_data.to_dict(orient="list")
print("- number of points: {}".format(len(senors_data['timestamp'])))
print("- start at - end at : {} - {}".format(
    senors_data['timestamp'][0], senors_data['timestamp'][-1]))
pps = group_point_per_sec(senors_data['timestamp'])
print("- mean, std point per second: {} , {}".format(np.mean(pps), np.std(pps)))

print("#"*100)
print("--FRAMES INDEX--")
print("- path :", args.frames_data_path)
print("- data frame csv:")
print(frames_index)
frames_index = frames_index.to_dict(orient="list")
print("- number of points: {}".format(len(frames_index['timestamp'])))
print("- start at - end at : {} - {}".format(
    frames_index['timestamp'][0], frames_index['timestamp'][-1]))
pps = group_point_per_sec(frames_index['timestamp'])
print("- mean, std point per second: {} , {}".format(np.mean(pps), np.std(pps)))



