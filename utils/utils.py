from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import asyncio
import websockets
import json
import threading


def show_animated_plot(fn_get_data, window_size=1000, freq=0.2, xlabel='Time', ylabel='Acc'):
    ''' Show animetd plot of updated each 'freq' calling 'fn_get_data' '''
    fig, ax = plt.subplots()
    data = fn_get_data()
    lines = [ax.plot(list(range(len(d)))[-window_size:], d)[0] for d in data]

    def update(frame):
        data = fn_get_data()
        if data:
            for i, d in enumerate(data):
                lines[i].set_data(list(range(len(d)))
                                  [-window_size:], d[-window_size:])
            # Update each 2 frames
            if frame % 2 == 0:
                ax.relim()
                ax.autoscale_view()
        fig.canvas.draw()
        return lines

    animation = FuncAnimation(
        fig, update, frames=len(data[0]), interval=freq*1000)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.canvas.draw()
    plt.show(block=False)
    return animation


def serve_websocket_data(fn_get_data, freq=0.2, host='localhost', port=8764):

    def background_serve_websocket(loop, ws, fn_get_data, freq=0.2, host='localhost', port=8764):
        print("Starting websocket server on port="+str(port))
        # start_server = websockets.serve(send_data, host, port)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ws)
        loop.run_forever()

    async def send_data(websocket, path):
        while True:
            data = fn_get_data()
            if data and len(data) > 0:
                await websocket.send(json.dumps(data))
                await asyncio.sleep(freq)

    loop = asyncio.new_event_loop()
    ws = websockets.serve(send_data, host, port, loop=loop)
    t = threading.Thread(target=background_serve_websocket, args=(
        loop, ws, fn_get_data, freq, host, port, ))
    t.setDaemon
    t.start()


class setInterval:
    ''' Run 'action' each 'interval' in async way. 
    Like javascript setInterval, see it to know behavior '''

    def __init__(self, interval, action):
        self.interval = interval
        self.action = action
        self.stopEvent = threading.Event()
        thread = threading.Thread(target=self.__setInterval)
        thread.start()

    def __setInterval(self):
        nextTime = time.time()+self.interval
        while not self.stopEvent.wait(nextTime-time.time()):
            nextTime += self.interval
            self.action()

    def cancel(self):
        self.stopEvent.set()