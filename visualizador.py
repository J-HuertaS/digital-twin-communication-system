import tkinter as tk
from tkinter import ttk
from collections import deque
from queue import Empty

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import Emisor  # 👈 para controlar SENSOR_LEVEL y CHANNEL_BER


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
        self.root.geometry("1100x780")

        # -------------------------
        # CONTENEDOR PRINCIPAL
        # -------------------------
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # -------------------------
        # PANEL DE CONTROLES
        # -------------------------
        controls = ttk.LabelFrame(main, text="Controles en tiempo real", padding=10)
        controls.pack(fill="x", pady=(0, 8))

        # Slider sensor
        self.sensor_var = tk.IntVar(value=Emisor.get_sensor_level())
        ttk.Label(controls, text="Nivel del sensor (ADC medio)").grid(row=0, column=0, sticky="w")
        sensor_slider = ttk.Scale(
            controls, from_=0, to=1023, orient="horizontal",
            variable=self.sensor_var,
            command=self._on_sensor_change
        )
        sensor_slider.grid(row=1, column=0, sticky="ew", padx=(0, 12))

        self.sensor_label = ttk.Label(controls, text=f"{self.sensor_var.get()} ADC")
        self.sensor_label.grid(row=1, column=1, sticky="w")

        # Slider BER
        self.ber_var = tk.DoubleVar(value=Emisor.get_channel_ber())
        ttk.Label(controls, text="Ruido del canal (BER)").grid(row=0, column=2, sticky="w")
        ber_slider = ttk.Scale(
            controls, from_=0.0, to=0.08, orient="horizontal",
            variable=self.ber_var,
            command=self._on_ber_change
        )
        ber_slider.grid(row=1, column=2, sticky="ew", padx=(0, 12))

        self.ber_label = ttk.Label(controls, text=f"{self.ber_var.get():.3f}")
        self.ber_label.grid(row=1, column=3, sticky="w")

        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(2, weight=1)

        tip = ttk.Label(
            controls,
            text="Tip demo: sube BER para ver cómo la reconstrucción se degrada.\n"
                 "Hamming(7,4) corrige 1 error por palabra: con BER alta quedan errores residuales.",
            justify="left"
        )
        tip.grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        # -------------------------
        # PANEL DE GRÁFICAS
        # -------------------------
        frame = ttk.Frame(main)
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

        self.status = ttk.Label(main, text="Esperando datos...")
        self.status.pack(anchor="w", pady=(6, 0))

    # -------------------------
    # CALLBACKS SLIDERS
    # -------------------------
    def _on_sensor_change(self, _=None):
        v = int(self.sensor_var.get())
        Emisor.set_sensor_level(v)
        self.sensor_label.config(text=f"{v} ADC")

    def _on_ber_change(self, _=None):
        v = float(self.ber_var.get())
        Emisor.set_channel_ber(v)
        self.ber_label.config(text=f"{v:.3f}")

    # -------------------------
    # CONSUMO DE COLAS
    # -------------------------
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

    # -------------------------
    # FFT
    # -------------------------
    def _fft_mag(self, x):
        if len(x) < 8:
            return None, None
        arr = np.array(x, dtype=float)
        arr = arr - np.mean(arr)
        N = len(arr)

        X = np.fft.rfft(arr)
        f = np.fft.rfftfreq(N, d=1 / self.fs)
        mag = np.abs(X)
        return f, mag

    # -------------------------
    # REDRAW
    # -------------------------
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
            text=f"Muestras | Emisor: {len(self.emitted)} | Receptor: {len(self.received)} | "
                 f"Fs≈{self.fs} Hz | Sensor={Emisor.get_sensor_level()} | BER={Emisor.get_channel_ber():.3f}"
        )

    def _tick(self):
        if self._consume():
            self._redraw()
        self.root.after(150, self._tick)
