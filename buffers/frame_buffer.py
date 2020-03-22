# Qui calsse camera buffer
import cv2
import time
import threading
import os


class FrameBuffer():
    """docstring for FrameBuffer"""

    def __init__(self, size, path=""):
        super().__init__()
        self.buffer = []
        self.size = size
        self.path = path
        self.condition = threading.Condition()

    def __push(self, frame):
        self.buffer.append(frame)

    def __pop(self):
        self.buffer.pop(0)

    def add(self, frame):
        with self.condition:
            if not self.condition.acquire():
                self.condition.wait()
            if len(self.buffer) > self.size:
                self.__pop()
            self.__push(frame)
            self.condition.release()
            self.condition.notifyAll()

    def __flush(self):
        self.buffer.clear()

    def pothole_detected(self):
        with self.condition:
            if not self.condition.acquire():
                self.condition.wait()
            print("swapping buffer ...")
            tmp_buffer = self.buffer
            self.tmp_buffer = []
            self.condition.release()
            self.condition.notifyAll()

        path = ("{}/".format(self.path))

        if not os.path.exists(path):
            os.makedirs(path)
        for f in self.buffer:
            frame = f.frame
            cv2.imwrite("{}{}.png".format(
                path, "{}".format(time.time())), frame)
        self.__flush()
