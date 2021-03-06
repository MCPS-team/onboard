from simulate import SimulateSensors, SimulateCamera
import numpy as np
from main import MainProcess
import threading
import argparse
from DeepESN_potholes import inference_all_data
from config import config
from threading import Timer

ANALYZED_FROM_SENSORS_PORT = 8761
INPUT_SENSORS_PORT = 8771

'''
Esempio di utilizzo:
Esegui:
python run_simulation.py -h per vedere i parametri
python run_simulation.py --data_path ./simulate/data/accelerometer_2020-03-06T111304369Z.csv --verbose

Per vedere anche la dimostrazione:
Apri ./monitor/index.html nel browser
python run_simulation.py --data_path ./simulate/data/accelerometer_2020-03-06T111304369Z.csv --speed 2 --verbose --monitor
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run onboard simulation test for RaspberryPi')
    parser.add_argument('--data_path',  type=str,
                        help='path of mocked data csv', required=True)
    parser.add_argument('--video_path',  type=str,
                        help='path of mocked video', required=True)
    parser.add_argument('--info_path',  type=str,
                        help='path of data [frame_index, timestamp]', required=True)
    parser.add_argument('--speed', type=float, default=1,
                        help='speed of simulation (es. 2 = 2x)')
    parser.add_argument('--verbose', action='store_true',
                        help="verbose mode, otherwise silent")
    parser.add_argument('--monitor', action='store_true',
                        help="serve incoming and analyzed data through webscoket. Use it for presentation.")
    parser.add_argument('--save_file', action='store_true',
                        help="Save generated potholes objects to json file.")

    args = parser.parse_args()

    print("Data path", args.data_path)
    main_process = MainProcess(config)
    sensor_simulation = SimulateSensors(
        args.data_path, freq=1/config.sensors_freqHz, speed=args.speed, verbose=args.verbose)
    camera_simulation = SimulateCamera(
        args.video_path, args.info_path, freq=1/config.camera_fps, speed=args.speed, verbose=args.verbose, preload=True)
    # def on_end():
    #     X = np.array([main_process.sensor_buffer.timeseries_history])
    #     y_pred = np.array(
    #         [main_process.sensor_buffer.timeseries_detected_history])
    #     print("IN DATA SHAPE", X.shape)
    #     print("OUT DATA SHAPE", y_pred.shape)
    #     # plt = plot_timeseries_clf(X, y_pred, transient=0)
    #     # plt.show()
    #     import pandas as pd
    #     pd.DataFrame({"predicted":y_pred.tolist()}).to_csv("predicted.csv",header=None, index=None)
    # sensor_simulation.on_end(on_end)

    # camera_simulation =
    sensor_simulation.run(main_process.on_update_sensors)
    camera_simulation.run(main_process.on_update_camera)

    if args.monitor:
        main_process.sensor_buffer.serve_websocket_data(
            freq=args.freq*2, port=ANALYZED_FROM_SENSORS_PORT)
        sensor_simulation.serve_websocket_data(
            freq=args.freq*2, port=INPUT_SENSORS_PORT)

    if args.save_file:
        import json
        # TODO: make better implementation
        def save_to_file(main_process, path):
            events = [event.to_dict() for event in main_process.sensor_buffer.events_history.history]
            with open(path, 'w') as f:
                json.dump(events, f)
            print("Saved {} event to file {}".format(len(events), path))
        r = Timer(5.0, save_to_file, (main_process, './log_pothole_events.json'))
        sensor_simulation.on_end(r.start)
