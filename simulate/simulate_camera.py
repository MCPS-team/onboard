# Qui classe Simulate Camera
# Vedi classe BaseSimulator per ereditare funzioni
# Una SetInterval come in sumilate sensors per simulare il flusso asincrono di dati

from .interface import BaseSimulation, setInterval


class SimulateCamera(BaseSimulation):
    def __init__(self, freq=0.2, speed=1, verbose=1):
        super().__init__(freq=freq, speed=speed, verbose=verbose)
        pass
