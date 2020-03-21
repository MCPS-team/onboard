# Qui classe Simulate Camera
# Vedi classe BaseSimulator per ereditare funzioni
# Una SetInterval come in sumilate sensors per simulare il flusso asincrono di dati

from .interface import BaseSimulation, setInterval
import cv2

class FrameWrapper():
    def __init__(self, frame, timestamp=None):
        super().__init__()
        self.frame = frame
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = time.time()

class SimulateCamera(BaseSimulation):
    def __init__(self, data_path,  video_buffer, freq=0.2, speed=1, verbose=1, ):
        super().__init__(freq=freq, speed=speed, verbose=verbose)
        self.data_path = data_path
        self.video_buffer = video_buffer
    
    def read_video(self, callback):
        cap = cv2.VideoCapture(self.data_path)
        if self.verbose:
            print("acquiring resource...")
            print(cap.isOpened())
        while(cap.isOpened()):  
            ret, frame = cap.read()
            if self.verbose:
                print ("frame red...")
            if ret==True:
                gray = cv2.cvtColor(frame, 0)

                frame_wrapped = FrameWrapper(frame) #timestamp corretto del mock)
                callback(frame_wrapped)

                cv2.imshow('frame',gray)

                time.sleep(self._freq)
            else:
                if self.verbose:
                    print("no frame read")
                break

        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()

