from .interface import PotholeEvent, PotholeEventHistory, SensorsData
from typing import List
import numpy as np
from DeepESN_potholes.inference import inference


class SensorsBuffer():
    def __init__(self, config={},  window_size: int = 200, detect_delay: int = 20, verbose: int = 0):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.detect_delay = detect_delay
        self.buffer = []
        self.events_history = PotholeEventHistory()
        self.timeseries_history = []
        self.timeseries_detected_history = np.zeros((self.window_size,))
        self.verbose = verbose
        # Pre-allocate matrix for performance
        self.__input_matrix = np.zeros((3, self.window_size))

    def append(self, data: SensorsData):
        ''' append accelarator data to the buffer.'''
        self.buffer.append(data)

    def is_ready(self):
        return (len(self.buffer) >= self.window_size and (len(self.buffer) % self.detect_delay == 0))

    def analyze(self, force: bool = False) -> List[PotholeEvent]:
        '''  Analyze data in last chunk "window_size" of buffer.
        Return a list of 'PotholeEvent' (empty list if no one pothole was detected, or buffer not ready).
        If force==True the chunk is always analyzed.
        If data in buffer is less then "window_size" the result might not be reliable.        
        '''
        events = []
        if len(self.buffer) >= self.window_size and (force or len(self.buffer) % self.detect_delay == 0):
            # copy_buffer = 
            for i, data in enumerate(self.buffer[-self.window_size:]):
                self.__input_matrix[0][i] = data.x
                self.__input_matrix[1][i] = data.y
                self.__input_matrix[2][i] = data.z

            events = self._detect(self.__input_matrix)
            # Add to history
            if len(events) > 0:
                for e in events:
                    self.events_history.append(e)
        return events

    def _detect(self, data) -> List[PotholeEvent]:
        out_timeseries = inference(data, self.config, verbose=self.verbose)[0]
        events = self.events_history.timeseries_to_events(
            self.buffer[-self.window_size:], out_timeseries)

        if self.verbose:
            if len(self.timeseries_history) > 0:
                self.timeseries_history = np.concatenate(
                    (self.timeseries_history, data[:, -self.detect_delay:]), axis=1)
                self.timeseries_detected_history = np.concatenate(
                    (self.timeseries_detected_history[:-(self.window_size-self.detect_delay)], out_timeseries), axis=0)
                # self.timeseries_detected_history[-self.window_size:] = out_timeseries
            else:
                self.timeseries_history = data
                self.timeseries_detected_history = np.array(out_timeseries)
        return events

    def serve_websocket_data(self, freq=0.2, port=8761):
        ''' start websocket server to send data to the monitor dashboard '''
        def get_data():
            data = [[], [], [], []]
            if len(self.timeseries_detected_history) > 0:
                data[0] = self.timeseries_history[0, :].tolist()
                data[1] = self.timeseries_history[1, :].tolist()
                data[2] = self.timeseries_history[2, :].tolist()
                data[3] = list(
                    map(lambda x: x*100, self.timeseries_detected_history.tolist()))
            return data
        return serve_websocket_data(get_data, port=port, freq=freq)

    def to_list(self) -> List[SensorsData]:
        return list(self.buffer)

    def save(self, path: str):
        pass


if __name__ == '__main__':
    import time
    fake_sensor_data = [[i, i, i] for i in range(1000)]
    buffer = SensorsBuffer()
    for d in fake_sensor_data:
        events = buffer.append(d[0], d[1], d[2])
        # Check if events is not None and is not empty
        if events is not None and len(events) > 0:
            for index, event in enumerate(events):
                print("Pothole event {}: start_at={} - end_at={}".format(index,
                                                                         event.start_at, event.end_at))
