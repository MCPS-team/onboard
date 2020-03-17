from detect_interface import SensorsData, PotholeEvent, SensorsBuffer
from config import config


class MainProcess():
    def __init__(self):
        super().__init__()
        self.sensor_buffer = SensorsBuffer(window_size=config.window_size, detect_delay=config.detect_delay, verbose=1)
        # self.camera_buffer = 

    def on_update_sensor(self, timestamp, x, y, z, lat=None, lng=None):
        acc_data = SensorsData(timestamp, x, y, z, lat, lng)
        self.sensor_buffer.append(acc_data)
        events = self.sensor_buffer.analyze()
        if events is not None:
            for event in events:
                    print("Pothole event {}: start_at={} - end_at={}".format(event.uid,
                                                                            event.start_at, event.end_at))

    def on_update_camera(self, ):
        pass




    





