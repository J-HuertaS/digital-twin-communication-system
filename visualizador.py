# visualizador.py
import tkinter as tk
from tkinter import ttk
from collections import deque
from queue import Empty

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LiveScope:
    def __init__(self, root, emit_q, recv_q, fs=50.0, max_samples=600):
        self.root = root
        self.emit_q = emit_q
        self.recv_q = recv_q
        self.fs = fs

        self.max_samples = max_samples
        self.emitted = deque(maxlen=max_samples)
        self.received = deque(maxlen=max_samples)

        self._build_ui()
        self._tick()

    def _build_ui(self):
        self.root.title("Monitor Voltajes + FFT (Emisor vs Receptor)")
        self.root.geometry("1100x720")

        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(9, 6))

        self.ax_e = self.fig.add_subplot(2, 2, 1)
        self.ax_r = self.fig.add_subplot(2, 2, 2)
        self.ax_fe = self.fig.add_subplot(2, 2, 3)
        self.ax_fr = self.fig.add_subplot(2, 2, 4)

        self.ax_e.set_title("Voltajes emitidos (filtrados)")
        self.ax_r.set_title("Voltajes recibidos (reconstruidos)")
        self.ax_fe.set_title("FFT voltajes emitidos")
        self.ax_fr.set_title("FFT voltajes recibidos")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.status = ttk.Label(frame, text="Esperando datos...")
        self.status.pack(anchor="w", pady=(6, 0))

    def _consume(self):
        updated = False

        # leer emisor
        while True:
            try:
                rep = self.emit_q.get_nowait()
            except Empty:
                break
            v = rep.get("emitted_volt", [])
            for x in v:
                self.emitted.append(float(x))
            updated = True

        # leer receptor
        while True:
            try:
                rep = self.recv_q.get_nowait()
            except Empty:
                break
            v = rep.get("received_volt", [])
            for x in v:
                self.received.append(float(x))
            updated = True

        return updated

    def _fft_mag(self, x):
        if len(x) < 8:
            return None, None
        arr = np.array(x, dtype=float)
        arr = arr - np.mean(arr)
        N = len(arr)

        X = np.fft.rfft(arr)
        f = np.fft.rfftfreq(N, d=1/self.fs)
        mag = np.abs(X)
        return f, mag

    def _redraw(self):
        self.ax_e.clear()
        self.ax_r.clear()
        self.ax_fe.clear()
        self.ax_fr.clear()

        self.ax_e.set_title("Voltajes emitidos (filtrados)")
        self.ax_r.set_title("Voltajes recibidos (reconstruidos)")
        self.ax_fe.set_title("FFT voltajes emitidos")
        self.ax_fr.set_title("FFT voltajes recibidos")

        if self.emitted:
            self.ax_e.plot(list(self.emitted))

        if self.received:
            self.ax_r.plot(list(self.received))

        fe, me = self._fft_mag(self.emitted)
        if fe is not None:
            self.ax_fe.plot(fe, me)

        fr, mr = self._fft_mag(self.received)
        if fr is not None:
            self.ax_fr.plot(fr, mr)

        self.canvas.draw_idle()

        self.status.config(
            text=f"Muestras mostradas | Emisor: {len(self.emitted)} | Receptor: {len(self.received)} | Fs≈{self.fs} Hz"
        )

    def _tick(self):
        if self._consume():
            self._redraw()
        self.root.after(150, self._tick)
