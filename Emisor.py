import asyncio
import json
import logging
import random
import numpy as np
import websockets
import serial
from queue import Queue

from Codificacion_Hamming import Hamming
from Codificacion_Huffman import train_codebook
from Codificacion_Huffman import encode as huf_encode
from Codificacion_Huffman import bits_to_bytes, bytes_to_bits  # No se usan, pero se mantienen
from Filtrado import apply_moving_average_filter, adc_to_voltage, voltage_to_adc, calculate_entropy


# =========================================================
# COLAS Y ESTADO COMPARTIDO PARA UI
# =========================================================
EMIT_Q = Queue(maxsize=50)

def get_emit_queue():
    return EMIT_Q

# Slider 1: nivel del "sensor" (ADC medio)
SENSOR_LEVEL = 520  # valor inicial
def set_sensor_level(v: int):
    global SENSOR_LEVEL
    SENSOR_LEVEL = int(v)

def get_sensor_level() -> int:
    return int(SENSOR_LEVEL)

# Slider 2: BER del canal
CHANNEL_BER = 0.00  # valor inicial
def set_channel_ber(v: float):
    global CHANNEL_BER
    CHANNEL_BER = float(v)

def get_channel_ber() -> float:
    return float(CHANNEL_BER)


# =========================================================
# CONFIGURACIÓN
# =========================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

USE_ARDUINO = False       # ← True si usan Arduino real
ARDUINO_PORT = "COM3"
ARDUINO_BAUD = 9600

WINDOW = 5
BLOCK_SIZE = 100

HOST = "localhost"
PORT = 8766  # 👈 IMPORTANTE: mismo puerto que tu Receptor


# =========================================================
# CANAL RUIDOSO (BER)
# =========================================================
def apply_ber(bits, ber):
    if ber <= 0:
        return bits
    noisy = []
    for b in bits:
        if random.random() < ber:
            noisy.append(1 - b)
        else:
            noisy.append(b)
    return noisy


# =========================================================
# HANDLER WEBSOCKET (EMISOR)
# =========================================================
async def handle_connection(websocket):
    logging.info("Cliente conectado al EMISOR")

    hamming = Hamming(k=4, n=7)
    buffer = []

    arduino = None
    if USE_ARDUINO:
        try:
            arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
            logging.info(f"Arduino conectado en {ARDUINO_PORT}")
        except Exception as e:
            logging.error(f"No se pudo abrir el puerto Arduino: {e}")
            return

    try:
        while True:
            # ------------------------------
            # LECTURA SEGÚN MODO
            # ------------------------------
            value = None

            if USE_ARDUINO and arduino:
                if arduino.in_waiting > 0:
                    raw = arduino.readline().decode('utf-8').strip()
                    if raw.isdigit():
                        value = int(raw)
                else:
                    await asyncio.sleep(0.001)
                    continue
            else:
                # 🔥 Sensor simulado controlado por SLIDER
                await asyncio.sleep(1 / 50)

                t = getattr(handle_connection, "_t", 0)
                handle_connection._t = t + 1

                base = get_sensor_level()          # 👈 desde slider
                slow = 90 * np.sin(2 * np.pi * 0.02 * t)
                noise = np.random.normal(0, 8)

                value = int(np.clip(base + slow + noise, 0, 1023))

            if value is None:
                continue

            buffer.append(value)

            if len(buffer) < BLOCK_SIZE:
                continue

            # ------------------------------
            # PROCESAR BLOQUE
            # ------------------------------
            block = np.array(buffer[:BLOCK_SIZE])
            buffer = buffer[BLOCK_SIZE:]

            entropy = calculate_entropy(block)

            # 1) ADC -> Voltaje
            volt = adc_to_voltage(block)

            # 2) Filtro promedio móvil
            filtered = apply_moving_average_filter(volt, WINDOW)

            # 3) Voltaje -> ADC (símbolos)
            filtered_adc = voltage_to_adc(filtered)

            # ------------------------------
            # REPORTE PARA VISUALIZACIÓN
            # (lo que realmente se transmite)
            # ------------------------------
            try:
                emitted_volt = adc_to_voltage(filtered_adc)
                EMIT_Q.put_nowait({
                    "filtered_adc": filtered_adc.tolist(),
                    "emitted_volt": emitted_volt.tolist(),
                    "entropy": float(entropy),
                    "ber": get_channel_ber()
                })
            except Exception:
                pass

            # ------------------------------
            # HUFFMAN
            # ------------------------------
            fuente = filtered_adc.tolist()
            codebook = train_codebook(fuente)
            encoded_huffman_bits_str = huf_encode(fuente, codebook)

            # ------------------------------
            # HAMMING
            # ------------------------------
            bits_list = [int(x) for x in encoded_huffman_bits_str]
            hamming_encoded_array = hamming.encode(bits_list)

            # ------------------------------
            # CANAL RUIDOSO (BER desde slider)
            # ------------------------------
            ber = get_channel_ber()
            hamming_noisy_list = apply_ber(hamming_encoded_array.tolist(), ber)

            # ------------------------------
            # ENVÍO
            # ------------------------------
            message = {
                "entropy": float(entropy),
                "codebook": codebook,
                "hamming": hamming_noisy_list,
                "huffman_length": len(bits_list),
                "channel_ber": ber
            }

            await websocket.send(json.dumps(message))

    except websockets.exceptions.ConnectionClosedOK:
        logging.info("Cliente desconectado del EMISOR (cierre normal)")
    except Exception as e:
        logging.error(f"Error en EMISOR: {e}", exc_info=True)
    finally:
        if arduino and arduino.is_open:
            arduino.close()
            logging.info("Puerto Arduino cerrado")


# =========================================================
# MAIN EMISOR (SERVIDOR)
# =========================================================
async def main():
    try:
        async with websockets.serve(handle_connection, HOST, PORT):
            logging.info(f"Servidor EMISOR listo en ws://{HOST}:{PORT}")
            await asyncio.Future()
    except Exception as e:
        logging.error(f"Error al iniciar el servidor: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
