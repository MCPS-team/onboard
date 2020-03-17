from detect_interface import SensorsData, onBoardPotholeEvent, SensorsBuffer
from config import config

buffer = SensorsBuffer(window_size=config.window_size, detect_delay=config.detect_delay, verbose=1)


def on_update_sensor(timestamp, x, y, z, lat=None, lng=None):
    acc_data = SensorsData(timestamp, x, y, z, lat, lng)
    buffer.append(acc_data)
    events = buffer.analyze()
    if events is not None:
        for event in events:
                print("Pothole event {}: start_at={} - end_at={}".format(event.uid,
                                                                         event.start_at, event.end_at))




    





