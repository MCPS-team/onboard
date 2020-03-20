from buffers import SensorsData, PotholeEvent, SensorsBuffer, CameraBuffer


class MainProcess():
    ''' Classe principale che racchide tutto il processo di analisi e raccolta dati '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sensor_buffer = SensorsBuffer(config=config, window_size=config.window_size, detect_delay=config.detect_delay, verbose=1)
        # self.camera_buffer = 

    def on_update_sensors(self, timestamp, x, y, z, lat=None, lng=None):
        acc_data = SensorsData(timestamp, x, y, z, lat, lng)
        self.sensor_buffer.append(acc_data)
        events = self.sensor_buffer.analyze()
        if events is not None:
            # Qui chiami la funzione che prende in input gli eventi e seleziona i fotogrammi.
            # e aggiungi i fotogrammi corrispondenti all'oggetto PotholeEvent
            for event in events:
                    print("Pothole event {}: start_at={} - end_at={}".format(event.uid,
                                                                            event.start_at, event.end_at))

    def on_update_camera(self, ):
        pass

    def on_near_depot(self, ):
        pass

    def upload_sensors_data(self, ):
        pass

    def upload_sensors_images(self, ):
        pass




    





