# Qui classe Simulate Camera
# Vedi classe BaseSimulator per ereditare funzioni
# Una SetInterval come in sumilate sensors per simulare il flusso asincrono di dati

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
    def __init__(self, data_path, info_path, freq=0.2, speed=1, verbose=1, ):
        super().__init__(freq=freq, speed=speed, verbose=verbose)
        self.data_path = data_path
        self.info_path = info_path
        self.df_info = pd.read_csv(self.info_path, names=[
                               'frame_index', 'timestamp'])

    def next_timestamp(self, index):
        if len(self.df_info['frame_index']) <= index:
            return None
        return self.df_info['timestamp'][index]

    def read_video(self, callback):
        cap = cv2.VideoCapture(self.data_path)
        if self.verbose:
            print("acquiring resource...")
            print(cap.isOpened())
        frame_index = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            timestamp = self.next_timestamp(frame_index)
            frame_index += 1
            if self.verbose:
                print("frame red...")
            if ret == True:
                gray = cv2.cvtColor(frame, 0)

                frame_wrapped = FrameWrapper(frame, timestamp)
                callback(frame_wrapped)

                # cv2.imshow('frame', gray)

                time.sleep(self._freq)
            else:
                if self.verbose:
                    print("no frame read")
                break

        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()

    def run(self, callback):
        if self.verbose:
            print("preparing t1...")
        self.thread = threading.Thread(
            name="reader", target=self.read_video, args=(callback,))
        self.thread.start()
        if self.verbose:
            print("reader started...")

