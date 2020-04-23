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
        return {'x': self.x, 'y': self.y, 'z': self.z, 'latitude': self.lat, 'longitude': self.lng,
                'timestamp': str(self.timestamp)}


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
        attached_sensor_data = []
        [attached_sensor_data.append(data.to_dict()) for data in self.attached_sensors_data]
        attached_frames = []
        [attached_frames.append({"filename": frame}) for frame in self.attached_images]
        return {"bumpID": str(self.uid),
                "latitude": self.latitude,
                "longitude": self.longitude,
                "start_at": str(self.start_at),
                "end_at": str(self.end_at),
                "attached_sensors_data": attached_sensor_data,
                "attached_images": attached_frames
                }


class PotholeEventHistory():
    ''' History of detected pothole events '''

    def __init__(self):
        super().__init__()
        self.history = []

    def append(self, event: PotholeEvent, check_last=True):
        if check_last:
            if len(self.history) > 0 and (
                    self.history[-1].overlap_event(event.start_at, event.end_at) or event.overlap_event(
                    self.history[-1].start_at, self.history[-1].end_at)):
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
                curr_event["end"] = index - 1
                new_event = PotholeEvent(
                    buffer_data[curr_event["start"]].timestamp, buffer_data[curr_event["end"]].timestamp)
                new_event.attached_sensors_data = buffer_data[curr_event["start"]: (curr_event["end"]+1)]
                new_event.latitude = buffer_data[curr_event["start"]].lat
                new_event.longitude = buffer_data[curr_event["start"]].lng
                events.append(new_event)
                curr_event = {"start": None, "end": None}
        # If is started but not ended discard event
        return events

    def to_list(self) -> List[PotholeEvent]:
        return self.history
