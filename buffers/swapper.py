import threading
import os

import cv2


class Swapper():
    def __init__(self, buffer, path=""):
        super().__init__()
        self.buffer = buffer
        self.thread = None
        self.path = path
        print("New swapper created...")

    def swap_and_save(self):
        old_buffer = self.buffer.pothole_detected()

        path = ("{}/".format(self.path))

        if not os.path.exists(path):
            os.makedirs(path)
        i = 0
        for f in old_buffer:
            frame = f.frame
            cv2.imwrite("{}{}.jpg".format(
                path, "{}".format(f.timestamp)), frame)
            print(f"Saved frame {f.timestamp} in filesystem!: {i} of {len(old_buffer)} ")
            i += 1

    def run(self):
        self.thread = threading.Thread(name="swapper", target =self.swap_and_save)
        self.thread.run()