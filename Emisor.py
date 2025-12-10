# emisor.py
import asyncio
import serial
import logging
import json
import base64
from time import sleep
from typing import List

import numpy as np

# Importar tus módulos locales
from Codificacion_Hamming import Hamming
from Codificacion_Huffman import train_codebook, encode as huffman_encode
from Filtrado import calculate_entropy, apply_moving_average_filter, adc_to_voltage, voltage_to_adc

# CONFIGURACIÓN - cámbialo según tu entorno
SERIAL_PORT = "COM3"           # <- Cambia a '/dev/ttyACM0' o el puerto correcto en Linux
BAUDRATE = 115200
BLOCK_SIZE = 100               # cada 100 datos creamos un bloque
FILTER_WINDOW = 5              # ventana promedio móvil
WS_URI = "ws://localhost:8765" # receptor debe estar escuchando aquí
Hamming_k = 4
Hamming_n = 7

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("emisor")

def bits_list_to_packed_bytes(bits: List[int]):
    """Empaqueta una lista de bits (0/1) en bytes. Devuelve (bytes, padding) donde padding es bits añadidos al final."""
    if not bits:
        return b"", 0
    s = "".join(str(b) for b in bits)
    padding = (8 - (len(s) % 8)) % 8
    s_padded = s + ("0" * padding)
    ba = bytearray()
    for i in range(0, len(s_padded), 8):
        byte = int(s_padded[i:i+8], 2)
        ba.append(byte)
    return bytes(ba), padding

def packed_bytes_to_bits_list(b: bytes, padding: int):
    if not b:
        return []
    bits = "".join(f"{byte:08b}" for byte in b)
    if padding:
        bits = bits[:-padding]
    return [int(ch) for ch in bits]

async def send_block_over_ws(payload: dict):
    import websockets
    # reconectar si falla
    while True:
        try:
            async with websockets.connect(WS_URI) as ws:
                await ws.send(json.dumps(payload))
                logger.info("Mensaje enviado por websocket.")
                # opcional: leer respuesta (ACK)
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    logger.info(f"Respuesta del receptor: {resp}")
                except asyncio.TimeoutError:
                    pass
            break
        except Exception as e:
            logger.warning(f"No pude conectar a {WS_URI}: {e}. Reintentando en 2s...")
            await asyncio.sleep(2.0)

def try_open_serial(port, baudrate):
    # intenta varios puertos comunes si falla
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        logger.info(f"Abierto serial en {port} @ {baudrate}")
        return ser
    except Exception as e:
        logger.warning(f"No pude abrir {port}: {e}")
        # no detener, lanzar la excepción para el llamador
        raise

async def main():
    # abrir puerto serial
    try:
        ser = try_open_serial(SERIAL_PORT, BAUDRATE)
    except Exception:
        logger.error("Asegúrate de poner SERIAL_PORT correcto en emisor.py")
        return

    buffer = []
    loop = asyncio.get_event_loop()

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                await asyncio.sleep(0.01)
                continue
            # suponemos que Arduino imprime un entero por línea
            try:
                val = int(line)
            except ValueError:
                logger.debug(f"Línea no es entero: {line!r}")
                continue

            buffer.append(val)
            if len(buffer) >= BLOCK_SIZE:
                block = buffer[:BLOCK_SIZE]
                del buffer[:BLOCK_SIZE]

                logger.info(f"=== BLOQUE listo ({len(block)} muestras) ===")
                logger.info(f"Bloque original: {block}")

                # Entropía
                ent = calculate_entropy(np.array(block))
                logger.info(f"Entropía Shannon del bloque: {ent:.6f} bits/símbolo")

                # Aplicar conversión ADC->voltaje->filtro->ADC para mantener símbolos enteros
                voltages = adc_to_voltage(np.array(block))
                filtered_volt = apply_moving_average_filter(voltages, FILTER_WINDOW)
                # apply_moving_average_filter produce array más corto (mode='valid'), ajustamos
                filtered_adc = voltage_to_adc(filtered_volt)
                filtered_adc_list = filtered_adc.tolist()
                logger.info(f"Bloque filtrado (ADC re-mapeado): {filtered_adc_list}")

                # Huffman (entrena con los símbolos filtrados)
                codebook = train_codebook(filtered_adc_list)
                logger.info(f"Diccionario Huffman: {codebook}")

                # Codificar con Huffman -> bits string
                try:
                    bits_string = huffman_encode(filtered_adc_list, codebook)  # string de '0'/'1'
                except Exception as e:
                    logger.error(f"Error codificando Huffman: {e}")
                    continue
                logger.info(f"Bloque codificado en Huffman (bits): {bits_string}")

                # Convertir a lista de bits ints para Hamming
                bits_list = [int(ch) for ch in bits_string]

                # Hamming
                h = Hamming(k=Hamming_k, n=Hamming_n)
                # Hamming.encode hará padding interno si es necesario y guarda data_length
                hamming_encoded = h.encode(bits_list)  # numpy array de bits
                hamming_bits_list = hamming_encoded.tolist()
                logger.info(f"Bloque codificado en Hamming (bits): {hamming_bits_list[:80]}{'...' if len(hamming_bits_list)>80 else ''}")
                logger.info(f"Hamming data_length (original bits length): {h.data_length}")

                # Empaquetar bits Hamming en bytes para enviar (y base64)
                hamming_bytes, hamming_padding = bits_list_to_packed_bytes(hamming_bits_list)
                hamming_b64 = base64.b64encode(hamming_bytes).decode()

                # Preparamos payload JSON
                payload = {
                    "type": "block",
                    "block_size": BLOCK_SIZE,
                    "entropy": ent,
                    "original_block": block,
                    "filtered_block": filtered_adc_list,
                    "huffman_codebook": codebook,
                    "huffman_bits_length": len(bits_list),
                    "hamming_k": Hamming_k,
                    "hamming_n": Hamming_n,
                    "hamming_data_length": h.data_length,  # longitud real de bits de Huffman (antes de padding)
                    "hamming_bits_base64": hamming_b64,
                    "hamming_bytes_padding": hamming_padding,
                    "hamming_bits_count": len(hamming_bits_list)
                }

                logger.info("Preparando a enviar payload por websocket (JSON).")
                await send_block_over_ws(payload)
                logger.info("Payload enviado; esperando siguiente bloque.")

        except Exception as e:
            logger.exception(f"Error en bucle principal del emisor: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Emisor detenido por usuario.")
