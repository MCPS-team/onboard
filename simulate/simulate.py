import pandas as pd
import threading
import time
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import serve_websocket_data
import sys
sys.path.insert(0, '../')


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
        ''' Return the data at next step, otherwise None'''
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


class SimulateSensors(BaseSimulation):
    def __init__(self, data_path, freq=0.2, speed=1, verbose=1):
        ''' Simulate Sensor in async way, reading mocked data from file and
        passing them to callback function with frequency=freq
        '''
        super().__init__(freq, speed, verbose)
        self.df = pd.read_csv(
            data_path,  names=['timestamp', 'x', 'y', 'z', 'target'])  # TODO add 'lat' 'lng'
        self.df['timestamp'] = self.df['timestamp'].apply(
            lambda iso_date: datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%S.%fZ"))
        self.df_index = 0

        if self.verbose:
            print("Data file length {}".format(len(self.df['timestamp'])))

    def get_data_until(self, index):
        ''' Return the data at next step, otherwise None'''
        if index >= len(self.df['timestamp']):
            index = len(self.df['timestamp']) - 1

        x = [self.df['timestamp'][:index], self.df['x'][:index],
             self.df['y'][:index], self.df['z'][:index]]
        return x

    def next_data(self):
        ''' Return the data at next step, otherwise None'''
        if self.df_index >= len(self.df['timestamp']):
            return None

        x = [self.df['timestamp'][self.df_index], self.df['x'][self.df_index],
             self.df['y'][self.df_index], self.df['z'][self.df_index]]
        self.df_index += 1
        return x

    def run(self, callback):
        ''' Start simulation. Run callback with frequency 'self.freq'
            with parameters the next step data.
            callback: function(timestamp, x, y, z, lat, lng)
        '''
        def next_step():
            if self.verbose:
                start_time = time.time()
            data = self.next_data()
            # if no more data, interrupt simulation and return
            if data is None:
                self.interval.cancel()
                if self.fn_on_end is not None:
                    self.fn_on_end()
                return False

            callback(data[0], data[1], data[2],
                     data[3])  # TODO add 'lat' 'lng'

            if self.verbose:
                diff = time.time()-start_time
                print("Running time for step {} : {} ms, required_time: {} ms, success: {}".format(
                    data[0], diff, self._freq, diff <= self._freq))
            return True

        self.interval = setInterval(self.freq, next_step)
        print(
            'Started simulated accelerometer input  -> time : {:.1f}s'.format(time.time()))

    def on_end(self, callback):
        self.fn_on_end = callback

    def serve_websocket_data(self, freq=0.2, port=8771, window_size=1000):
        ''' start websocket server to send data to the monitor dashboard '''
        def get_data():
            data = [[], [], []]
            sensors_data = self.get_data_until(self.df_index)
            if sensors_data:
                # temporal_index = list(range(self.df_index))[-window_size:]
                data[0] = sensors_data[1][-window_size:].tolist()
                data[1] = sensors_data[2][-window_size:].tolist()
                data[2] = sensors_data[3][-window_size:].tolist()
            return data
        return serve_websocket_data(get_data, port=port, freq=freq)


class SimulateCamera(BaseSimulation):
    def __init__(self, freq=0.2, speed=1, verbose=1):
        super().__init__(freq=freq, speed=speed, verbose=verbose)
        pass


if __name__ == '__main__':
    sim = SimulateSensors(
        './data/accelerometer_2020-03-06T111304369Z.csv', freq=0.2, speed=10, verbose=1)

    def process(timestamp, x, y, z):
        print("--> Callback", timestamp, x, y, z)

    sim.run(process)
