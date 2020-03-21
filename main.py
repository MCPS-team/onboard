from buffers import SensorsData, PotholeEvent, SensorsBuffer, FrameBuffer


class MainProcess():
    ''' Classe principale che racchide tutto il processo di analisi e raccolta dati '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sensor_buffer = SensorsBuffer(
            config=config, window_size=config.window_size, detect_delay=config.detect_delay, verbose=1)
        self.frame_buffer = FrameBuffer(config.camera_buffer_size, "./frames/")
        # self.camera_buffer =

    def on_update_sensors(self, timestamp, x, y, z, lat=None, lng=None):
        acc_data = SensorsData(timestamp, x, y, z, lat, lng)
        self.sensor_buffer.append(acc_data)
        events = self.sensor_buffer.analyze()
        if events is not None:
            # Qui chiami la funzione che prende in input gli eventi e seleziona i fotogrammi.
            # e aggiungi i fotogrammi corrispondenti all'oggetto PotholeEvent
            for event in events:
                self.frame_buffer.poth_hole_detected()
                print("Pothole event {}: start_at={} - end_at={}".format(event.uid,
                                                                         event.start_at, event.end_at))
        # In base a latitudine e longitudine, controlla
        # se siamo nelle vicinanze del deposito
        # self.on_near_depot

    def on_update_camera(self, frame):
        self.frame_buffer.add(frame)

    def on_near_depot(self, ):
        # quando nelle vicinanze invoca funzione per collegarsi al wifi
        # se torna True allora è collegato al wifi altrimenti ci riprova ogni tot secondi
        # self.upload_sensors_data()
        # self.upload_frames()
        pass

    def upload_sensors_data(self, ):
        pass

    def upload_frames(self, ):
        pass

