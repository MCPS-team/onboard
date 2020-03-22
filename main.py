from buffers import SensorsData, PotholeEvent, SensorsBuffer, FrameBuffer
from utils import euclidean_dist, setInterval
import time


class MainProcess():
    ''' Classe principale che racchide tutto il processo di analisi e raccolta dati '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sensor_buffer = SensorsBuffer(
            config=config, window_size=config.window_size, detect_delay=config.detect_delay, verbose=1)
        self.frame_buffer = FrameBuffer(config.camera_buffer_size, "./frames/")
        self.last_timestap = time.time()
        self.checking_edge_connection = None
        # self.camera_buffer =

    def on_update_sensors(self, timestamp, x, y, z, lat=None, lng=None):
        self.last_timestap = timestamp
        acc_data = SensorsData(timestamp, x, y, z, lat, lng)
        self.sensor_buffer.append(acc_data)
        events = self.sensor_buffer.analyze()
        if events is not None:
            # Qui chiami la funzione che prende in input gli eventi e seleziona i fotogrammi.
            # e aggiungi i fotogrammi corrispondenti all'oggetto PotholeEvent
            for event in events:
                self.frame_buffer.pothole_detected()
                print("Pothole event {}: start_at={} - end_at={}".format(event.uid,
                                                                         event.start_at, event.end_at))
        # In base a latitudine e longitudine, controlla
        # se siamo nelle vicinanze del deposito
        self.is_near_depot([lat, lng])

    def on_update_camera(self, frame):
        self.frame_buffer.add(frame)

    def is_near_depot(self, lat_lng):
        ''' Calcola distanza dal deposito in base a lat e lng,
        se inferiore a  config.depot_radius ogni due secondi 
        prova a connettersi al wifi chiamando try_wireless_connection.
        Se si allontata interrompe l'azione
        '''
        dist = euclidean_dist(lat_lng, self.config.depot_location)
        print(dist, self.config.depot_radius)
        # 0.00001 degrees is about a meter
        if dist <= self.config.depot_radius and not self.checking_edge_connection:
            self.checking_edge_connection = setInterval(self.config.retry_connection_delay, self.try_wireless_connection)
        elif dist > self.config.depot_radius and self.checking_edge_connection:
            self.checking_edge_connection.cancel()

    def try_wireless_connection(self):
        # Wrapper per connesione wifi
        # Chiama on on_edge_connection se riesce a collegarsi
        # Quando è connesso chiama self.checking_edge_connection.cancel() per non provare più a connettersi 
        # e poi chiama on_edge_connection
        print("!!!--In deposito--!!!"*5)
        self.checking_edge_connection.cancel()
        return

    def on_edge_connection(self):
        # quando connesso al wifi edge
        self.upload_sensors_data()
        self.upload_frames()

    def upload_sensors_data(self, ):
        pass

    def upload_frames(self, ):
        pass
