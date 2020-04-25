import pandas as pd
import threading
import time
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import serve_websocket_data


# Usato per simulare fps perchè lanica lazione ogni "interval" secondi
class setInterval:
    ''' Run 'action' each 'interval' in async way. 
    Like javascript setInterval, see it to know behavior '''

    def __init__(self, interval, action):
        self.interval = interval
        self.action = action
        self.stopEvent = threading.Event()
        thread = threading.Thread(target=self.__setInterval)
        thread.start()

    def __setInterval(self):
        nextTime = time.time()+self.interval
        while not self.stopEvent.wait(nextTime-time.time()):
            nextTime += self.interval
            self.action()


    def cancel(self):
        self.stopEvent.set()

# Usa class base per creare modificare SimulateCamera
# Prendi ispirazione a SimulateSensors

class BaseSimulation:
    def __init__(self, freq=0.2, speed=1, verbose=1):
        self._freq = freq
        self.freq = freq/speed
        self.interval = None
        self.verbose = verbose
        self.fn_on_end = None

    def get_data_until(self, index):
        ''' Return all data until step "index"'''
        pass

    def next_data(self):
        ''' Return the data at next step, otherwise None'''
        pass

    def run(self, callback):
        ''' Start simulation. Run callback with frequency 'self.freq'
            with parameters the next step data.
        '''
        pass

    def on_end(self, callback):
        ''' Call on end of simulation, when data are terminated '''
        pass

    def serve_websocket_data(self, freq=0.2, port=8771, window_size=1000):
        ''' start websocket server to send data to the monitor dashboard '''
        pass
