import numpy as np
from DeepESN_potholes import inference
from typing import List
import uuid
from utils import serve_websocket_data


class SensorsData():
    ''' Interface for data obtained from accelerator sensor. '''

    def __init__(self, timestamp: int, x: float, y: float, z: float, lat: float, lng: float):
        super().__init__()
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.lat = lat
        self.lng = lng

    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'z': self.z, 'lat': self.lat, 'lgn': self.lng, 'timestamp': self.timestamp}


class PotholeEvent():
    ''' Interface for event that identify a pothole, with attached data.

    Parameters:
    start_at = initial timestamp of detected pothole
    end_at = ending timestamp of detected pothole
    latitude = gps latitude assigned to pothole
    longitude = gps latitude assigned to pothole
    attached_sensors_data = List[SensorData] occurred during PotholeEvent
    attached_images = List[string] path or uid of images (to be defined) captured during PotholeEvent
    attached_video = List[string] path or uid of video captured during PotholeEvent
    '''

    def __init__(self, start_at: int, end_at: int):
        super().__init__()
        self.uid = uuid.uuid4()
        self.start_at = start_at
        self.end_at = end_at
        self.latitude = None
        self.longitude = None
        self.attached_sensors_data = []
        self.attached_images = []
        self.attached_video = []

    def contain_event(self, start_at: int, end_at: int) -> bool:
        ''' Contain timestamp defined by start_at, end_at (less or equal). '''
        return self.start_at <= start_at and end_at <= self.end_at

    def overlap_event(self, start_at: int, end_at: int) -> bool:
        ''' Overlap timestamp defined by start_at, end_at (less or equal). '''
        overlap_left = self.start_at >= start_at and start_at <= self.end_at
        overlap_left = self.start_at <= end_at and end_at <= self.end_at
        contain_in = self.contain_event(start_at, end_at)
        return overlap_left or overlap_left or contain_in

    def to_dict(self):
        pass


class PotholeEventHistory():
    ''' History of detected pothole events '''

    def __init__(self):
        super().__init__()
        self.history = []

    def append(self, event: PotholeEvent, check_last=True):
        if check_last:
            if len(self.history) > 0 and (self.history[-1].overlap_event(event.start_at, event.end_at) or event.overlap_event(self.history[-1].start_at, self.history[-1].end_at)):
                self.history.pop()
        self.history.append(event)

    def timeseries_to_events(self, buffer_data: List[SensorsData], pothole_timeseries: List[int]) -> List[PotholeEvent]:
        events = []
        curr_event = {"start": None, "end": None}
        for index, t in enumerate(pothole_timeseries):
            if index >= len(buffer_data):
                break
            if t == 1 and curr_event["start"] is None:
                curr_event["start"] = index
            if t != 1 and curr_event["start"] is not None:
                curr_event["end"] = index-1
                new_event = PotholeEvent(
                    buffer_data[curr_event["start"]].timestamp, buffer_data[curr_event["end"]].timestamp)
                new_event.attached_sensors_data = buffer_data[curr_event["start"]                                                              : curr_event["end"]]
                events.append(new_event)
                curr_event = {"start": None, "end": None}
        # If is started but not ended discard event
        return events

    def to_list(self) -> List[PotholeEvent]:
        return self.history

class SensorsBuffer():
    def __init__(self, window_size: int = 100, detect_delay: int = 10, verbose: int = 0):
        super().__init__()
        self.window_size = window_size
        self.detect_delay = detect_delay
        self.buffer = []
        self.events_history = PotholeEventHistory()
        self.timeseries_history = []
        self.timeseries_detected_history = []
        self.verbose = verbose
        # Pre-allocate matrix for performance
        self.__input_matrix = np.zeros((3, self.window_size))

        # if self.verbose>1:
        #     self.serve_websocket_sensors_data(freq=0.2)

    def append(self, data: SensorsData):
        ''' append accelarator data to the buffer.'''
        self.buffer.append(data)

    def analyze(self, force: bool = False) -> List[PotholeEvent]:
        '''  Analyze data in last chunk "window_size" of buffer.
        Return a list of 'PotholeEvent' (empty list if no one pothole was detected, or buffer not ready).
        If force==True the chunk is always analyzed.
        If data in buffer is less then "window_size" the result might not be reliable.        
        '''
        events = []
        if len(self.buffer) >= self.window_size and (force or len(self.buffer) % self.detect_delay == 0):
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
        out_timeseries = inference(data, verbose=self.verbose)[0]
        events = self.events_history.timeseries_to_events(
            self.buffer[-self.window_size:], out_timeseries)

        if self.verbose:
            if len(self.timeseries_detected_history) >= self.window_size:
                print(len(self.timeseries_history[0,:]), len(self.timeseries_history[0,:])+len(data[0, -self.detect_delay:]), len(self.timeseries_history[0,:])+self.detect_delay)
                self.timeseries_history = np.concatenate((self.timeseries_history, data[:, -self.detect_delay:]), axis=1)
                self.timeseries_detected_history = np.concatenate((self.timeseries_detected_history, np.zeros((self.detect_delay,))), axis=0)
                self.timeseries_detected_history[-self.window_size:] = out_timeseries
            else:
                self.timeseries_history = data
                self.timeseries_detected_history = out_timeseries
        return events

    def serve_websocket_data(self, freq=0.2, port=8761):
        ''' start websocket server to send data to the monitor dashboard '''
        def get_data():
            data = [[], [], [], []]
            if len(self.timeseries_detected_history)>0:
                data[0] = self.timeseries_history[0, :].tolist()
                data[1] = self.timeseries_history[1, :].tolist()
                data[2] = self.timeseries_history[2, :].tolist()
                data[3] = list(map(lambda x: x*100, self.timeseries_detected_history.tolist()))
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
