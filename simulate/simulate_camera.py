# Qui classe Simulate Camera
# Vedi classe BaseSimulator per ereditare funzioni
# Una SetInterval come in sumilate sensors per simulare il flusso asincrono di dati
import datetime

from .interface import BaseSimulation, setInterval
import cv2
import threading
import pandas as pd
import time

class FrameWrapper():
    def __init__(self, frame, timestamp=None):
        super().__init__()
        self.frame = frame
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = time.time()
        self.thread = None


class SimulateCamera(BaseSimulation):
    def __init__(self, video_path, info_path, freq=0.2, speed=1, verbose=1, preload=True):
        super().__init__(freq=freq, speed=speed, verbose=verbose)
        self.video_path = video_path
        self.info_path = info_path
        self.preload = preload
        self.df_info = pd.read_csv(self.info_path, names=[
                                'timestamp', 'frame_index'])
        if (self.preload):
            self.cached_frames = []
            self.preload_video()


    def next_timestamp(self, index):
        if len(self.df_info['frame_index']) <= index:
            return None
        return self.df_info['timestamp'][index]

    def preload_video(self):
        if(self.verbose):
            print("Loading video to cache")
        cap = cv2.VideoCapture(self.video_path)
        frame_index = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                timestamp = self.next_timestamp(frame_index)
                frame_wrapped = FrameWrapper(frame, timestamp)
                self.cached_frames.append(frame_wrapped)
                frame_index += 1
            else:
                break
        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()

    def read_video(self, callback):
        cap = cv2.VideoCapture(self.video_path)
        frame_index = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            timestamp = self.next_timestamp(frame_index)
            frame_index += 1
            if ret == True:
                self.frames.append(FrameWrapper(frame, timestamp))
            else:
                break
        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()
    
    def read_cache(self, callback):
        for frame in self.cached_frames:
            if self.verbose:
                print(f"frame {frame.timestamp} red...")
            callback(frame)
            time.sleep(self._freq)
        if self.verbose:
                    print("no frame read")

    def next_timestamp(self, index):
        if len(self.df_info['frame_index']) <= index:
            return None
        return self.df_info['timestamp'][index]

    def read_video(self, callback):
        for frame_wrapped in self.frames:
            if self.verbose:
                print(f"frame {frame_wrapped.timestamp} red at {time.time()}")

            callback(frame_wrapped)

            #mock
            time.sleep(self._freq)
        if self.verbose:
            print("no frame read")


    def run(self, callback):
        if self.verbose:
            print("preparing t1...")
        if self.preload:
            self.thread = threading.Thread(
                name="reader", target=self.read_cache, args=(callback,))
        else:
             self.thread = threading.Thread(
                name="reader", target=self.read_video, args=(callback,))
        self.thread.start()
        if self.verbose:
            print("reader started...")

