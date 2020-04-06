import argparse
import cv2
import pandas as pd
from time import time
from datetime import datetime
from datetime import timedelta
import numpy as np
import random
import PIL
from PIL import Image
import sys


def make_square(im, size=256, fill_color=(0, 0, 0, 0)):
    im = Image.fromarray(
        cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    )
    im.thumbnail((size,size), Image.ANTIALIAS)
    x, y = im.size
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return cv2.cvtColor(np.array(new_im), cv2.COLOR_RGB2BGR) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cut video to fit with the given data [info_path]')
    parser.add_argument('--sensors_path',  type=str,
                        help='path of sensors data', required=True)
    parser.add_argument('--video_path',  type=str,
                        help='path of mocked video', required=True)
    parser.add_argument('--fps',  type=int,
                        help='desired fps from video', required=True)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    take_frame_by = int(round(original_fps/args.fps))
    time_perturbation_ratio = (1/args.fps)*int(random.randrange(100, 300)/100)
    print("#"*30)
    print("Original Video")
    print("-"*30)

    print(pd.DataFrame({"path": [args.video_path], "fps": [original_fps]}))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('out_video.avi', fourcc, float(args.fps-time_perturbation_ratio), (800, 800), True)

    df_sensors = pd.read_csv(args.sensors_path, names=[
        'timestamp', 'x', 'y', 'z', 'lat', 'lng', 'target'])
    len_df_sensors = len(df_sensors.index)
    info_path = {'timestamp': [], 'frame_index': []}
    start = time()
    # Check if camera opened successfully

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    frame_counter = 0
    frame_index = 0
   

    start_timestamp = datetime.strptime(
        df_sensors['timestamp'].iloc[0], "%Y-%m-%dT%H:%M:%S.%fZ")
    end_timestamp = datetime.strptime(
        df_sensors['timestamp'].iloc[-1], "%Y-%m-%dT%H:%M:%S.%fZ")

    print("Sensors file start_at {}, end_at {}".format(start_timestamp, end_timestamp))
    timestamp = start_timestamp

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            if frame_counter % take_frame_by == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
                # Display the resulting frame
                info_path['frame_index'] += [frame_index]
                info_path['timestamp'] += [
                    timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")]
                frame = make_square(frame, size=800)
                out.write(frame)
                # cv2.imshow('Frame', frame)

                frame_index += 1

                time_perturbation = random.randrange(int(-time_perturbation_ratio*100), int(time_perturbation_ratio*100))/100
                timestamp = timestamp + timedelta(seconds=(1/args.fps)+time_perturbation) 

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                if timestamp > end_timestamp:
                    break
        # Break the loop
        else:
            break
        frame_counter += 1

    if len_df_sensors > frame_index:
        print("!!!Warning!!! Data length {} not reached. Video length = {}".format(
            len_df_sensors, frame_index))

    # format to 3 decimal seconds
    info_path['timestamp'] = [str_time[:23]+'Z' for str_time in info_path['timestamp']]
    
    df_info = pd.DataFrame(info_path)
    df_info.to_csv("frame_index.csv", header=False, index=False)
    print(df_info)

    # print("Video obtained with fps {}".format(out.get(cv2.CAP_PROP_FPS)))

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
