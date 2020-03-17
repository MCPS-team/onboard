from simulate import SimulateSensors #,SimulateCamera
import numpy as np
from main import on_update_sensor, buffer
from utils import plot_timeseries_clf
import threading
import argparse

ANALYZED_FROM_SENSORS_PORT = 8761
INPUT_SENSORS_PORT = 8771

# For local testing
def on_end():
        real_data = np.array([sensor_simulation.get_data_until(100000)])[:,1:,:]
        fake_class = np.zeros((1, real_data.shape[-1]))
        print("INPUT DATA SHAPE", real_data.shape)
        X =  np.array([buffer.timeseries_history])
        y_pred = np.array([buffer.timeseries_detected_history])
        print("DETECT DATA SHAPE", X.shape)
        plt = plot_timeseries_clf(real_data, fake_class, transient=0)
        plt.show()
        plt = plot_timeseries_clf(X, y_pred, transient=0)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run onboard simulation test for RaspberryPi')
    parser.add_argument('--data_path',  type=str, nargs=1, help='path of mocked data csv')
    parser.add_argument('--freq', type=float, default=0.2, help='frequency of obtained data from sensors')
    parser.add_argument('--speed', type=float, default=1, help='speed of simulation (es. 2 = 2x)')  
    parser.add_argument('--verbose', action='store_true', help="verbose mode, otherwise silent") 
    parser.add_argument('--monitor', action='store_true', help="serve incoming and analyzed data through webscoket. Use it for presentation.") 

    args = parser.parse_args()

    print("Data path", args.data_path[0])
    sensor_simulation = SimulateSensors(args.data_path[0], freq=args.freq, speed=args.speed, verbose=args.verbose)
    # sensor_simulation.on_end(on_end)
    sensor_simulation.run(on_update_sensor)
    if args.monitor:
        buffer.serve_websocket_data(freq=args.freq*2, port=ANALYZED_FROM_SENSORS_PORT)
        sensor_simulation.serve_websocket_data(freq=args.freq*2, port=INPUT_SENSORS_PORT)
    # camera_simulation = 
