import datetime
import os
import shutil
from datetime import timedelta
import requests
import config
from buffers import SensorsData, SensorsBuffer, FrameBuffer, Swapper
from utils import euclidean_dist, setInterval
import time
from dotenv import load_dotenv

# TODO: aggiungere thread per analisi rete neurale e salvataggio immagini
# TODO: salvare immagini in jpg
# TODO: implementare speed in simulazione fotocamera

class MainProcess():
    ''' Classe principale che racchide tutto il processo di analisi e raccolta dati '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sensor_buffer = SensorsBuffer(
            config=config, window_size=config.window_size, detect_delay=config.detect_delay, verbose=1)
        self.frame_buffer = FrameBuffer(config.camera_buffer_size)
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
                swapper = Swapper(self.frame_buffer, config.config.frames_path)
                swapper.run()
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
        # 0.00001 degrees is about a meter
        if dist <= self.config.depot_radius and not self.checking_edge_connection:
            self.checking_edge_connection = setInterval(self.config.retry_connection_delay, self.try_wireless_connection)
        elif dist > self.config.depot_radius and self.checking_edge_connection:
            self.checking_edge_connection.cancel()

    def try_wireless_connection(self):
        load_dotenv()
        SSID = os.getenv('SSID')
        PSW = os.getenv('SSID_PSW')
        os.system(f'nmcli device wifi con "{SSID}" password "{PSW}"')

        #TODO mokkare tentativo di connessione in modo da capire se si connette o meno e spostare tutto questo sotto in un blocco if
        print("connecting to edge server...")
        self.checking_edge_connection.cancel()
        self.on_edge_connection()

        return

    def try_connection(self):
        return True

    def on_edge_connection(self):
        # quando connesso al wifi edge
        attached_frames = self.attach_frames()
        self.upload_pothole_events()
        self.upload_frames(attached_frames)

    def attach_frames(self):
        frames = list(os.walk(config.config.frames_path))[0][2]
        attached_frames = []

        #clean folder from hidden files
        [frames.remove(x) for x in frames if x.find(".jpg") == -1]

        frames.sort()

        for event in self.sensor_buffer.events_history.history:
            start = event.start_at - timedelta(seconds=1)
            finish = event.end_at - timedelta(seconds=1)

            for frame in frames:
                ts = datetime.datetime.strptime(frame.split('.jpg')[0].replace('T', ' ').replace('Z', '').replace('/',':'), '%Y-%m-%d %H:%M:%S.%f')

                if start <= ts <= finish:
                    event.attached_images.append(frame)
                    attached_frames.append(frame)

        return attached_frames

    def upload_pothole_events(self, ):
        # Upload potholes_event objects
        events = []
        [events.append(event.to_dict()) for event in self.sensor_buffer.events_history.history]
        payload = {"data": events}
        
        print("sending sensor data...")
        r = requests.post("http://{}:{}/api/upload/bump-data".format(config.config.edge_ip, config.config.edge_port), json=payload)
        print(payload)
        if r.status_code == 200:
            print("Data upload successfully!")
        return

    def upload_frames(self, attached_frames ):
        files = []
        [files.append((file,open("{}/{}".format(config.config.frames_path, file), 'rb'))) for file in attached_frames]
        requests.post("http://{}:{}/api/upload/images".format(config.config.edge_ip, config.config.edge_port), files=files) 
        print("frames sent successfully!")

        for filename in os.listdir(config.config.frames_path):
            file_path = os.path.join(config.config.frames_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        return

