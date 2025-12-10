import asyncio
import json
import logging
import random
import numpy as np
import websockets
import serial

from Codificacion_Hamming import Hamming
from Codificacion_Huffman import train_codebook
from Codificacion_Huffman import encode as huf_encode
from Codificacion_Huffman import bits_to_bytes, bytes_to_bits
from Filtrado import apply_moving_average_filter, adc_to_voltage, voltage_to_adc, calculate_entropy
from queue import Queue

EMIT_Q = Queue(maxsize=50)

def get_emit_queue():
    return EMIT_Q

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configuración de E/S y Procesamiento
USE_ARDUINO = False        # ← CAMBIAR ESTO A True PARA LEER DEL ARDUINO
ARDUINO_PORT = "COM3"
ARDUINO_BAUD = 9600

WINDOW = 5
BLOCK_SIZE = 100


async def handle_connection(websocket):
    logging.info("Cliente conectado al EMISOR")

    # Inicialización de Hamming
    # Se inicializa Hamming(7,4) por defecto
    hamming = Hamming(k=4, n=7) 
    buffer = []
    
    # El diccionario de Huffman puede cambiar en cada bloque, 
    # por eso no se pre-calcula.

    # Abrir puerto serial solo si el usuario lo pidió
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
            # LECTURA SEGÚN MODO SELECCIONADO
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
                # Datos simulados
                await asyncio.sleep(1 / 200)
                t = getattr(handle_connection, "_t", 0)
                handle_connection._t = t + 1

                base = 520
                slow = 90 * np.sin(2 * np.pi * 0.02 * t)
                noise = np.random.normal(0, 8)
                value = int(np.clip(base + slow + noise, 0, 3000))

            
            if value is None:
                continue

            buffer.append(value)

            if len(buffer) < BLOCK_SIZE:
                continue

            # Procesar bloque
            block = np.array(buffer[:BLOCK_SIZE])
            buffer = buffer[BLOCK_SIZE:]

            logging.info(f"Bloque de datos sin procesar: {block.tolist()}")

            # ------------------------------
            # FILTRADO Y PROCESAMIENTO
            # ------------------------------
            
            # Calcular la entropía antes de filtrar (opcional, pero puede ser útil)
            entropy = calculate_entropy(block)
            logging.info(f"Entropía del bloque (crudo): {entropy}")

            # 1. Aplicar filtro de promedio móvil (en dominio de voltaje)
            volt = adc_to_voltage(block)
            filtered = apply_moving_average_filter(volt, WINDOW)
            
            # 2. Convertir de nuevo a valores ADC discretos (que son los símbolos)
            filtered_adc = voltage_to_adc(filtered)
            
            # El filtro de promedio móvil reduce la longitud del array
            # mode='valid' resulta en len(data) - window + 1
            logging.info(f"Bloque filtrado ADC ({len(filtered_adc)} samples): {filtered_adc.tolist()}")

            # ----------------------------------
            # REPORTE PARA VISUALIZACIÓN
            # ----------------------------------
            try:
                # Esta es la señal que realmente se transmite como símbolos
                emitted_volt = adc_to_voltage(filtered_adc)

                EMIT_Q.put_nowait({
                    "filtered_adc": filtered_adc.tolist(),
                    "emitted_volt": emitted_volt.tolist()
                })
            except Exception:
                pass

            # ------------------------------
            # CODIFICACIÓN DE HUFFMAN
            # ------------------------------


            # Los símbolos para Huffman deben ser los valores ADC discretos (enteros)
            # El script original convertía a str(filtered_adc.tolist()), lo cual
            # codificaba los caracteres de la representación de la lista, NO los valores ADC.
            
            # Usar los valores ADC discretos como fuente de símbolos
            fuente = filtered_adc.tolist() # Lista de enteros (símbolos)
            
            # Entrenar el codebook con los símbolos
            codebook = train_codebook(fuente)
            
            # Codificar la secuencia de símbolos en una cadena de bits
            encoded_huffman_bits_str = huf_encode(fuente, codebook)

            logging.info(f"Diccionario de Huffman: {codebook}")
            logging.info(f"Bits Huffman codificados (longitud {len(encoded_huffman_bits_str)}): {encoded_huffman_bits_str[:]}...")

            # ------------------------------
            # CODIFICACIÓN DE HAMMING
            # ------------------------------

            # 1. Convertir la cadena de bits de Huffman a una lista/array de enteros (0s y 1s)
            # El script original tenía una doble codificación y un uso de variable incorrecto.
            bits_list = [int(x) for x in encoded_huffman_bits_str]
            
            # 2. Codificar con Hamming
            # hamming.encode espera una lista/array de enteros (0s y 1s)
            hamming_encoded_array = hamming.encode(bits_list)

            logging.info(f"Hamming codificado (longitud {len(hamming_encoded_array)}): {hamming_encoded_array.tolist()[:]}...")

            # ------------------------------
            # ENVÍO
            # ------------------------------

            message = {
                "entropy": entropy,
                "codebook": codebook,
                # Se envían los bits codificados por Hamming como una lista de enteros
                "hamming": hamming_encoded_array.tolist(),
                # Longitud original de los bits Huffman antes del padding de Hamming
                "huffman_length": len(bits_list)
            }

            await websocket.send(json.dumps(message))
            logging.info(f"Mensaje enviado (tamaño {len(json.dumps(message))} bytes)")

    except websockets.exceptions.ConnectionClosedOK:
        logging.info("Cliente desconectado del EMISOR (cierre normal)")
    except Exception as e:
        logging.error(f"Error en EMISOR: {e}", exc_info=True)

    finally:
        if arduino and arduino.is_open:
            arduino.close()
            logging.info("Puerto Arduino cerrado")


async def main():
    try:
        async with websockets.serve(
            handle_connection,
            "localhost",
            8766
        ):
            logging.info("Servidor EMISOR listo en ws://localhost:8766")
            await asyncio.Future()  # Mantener vivo
    except Exception as e:
        logging.error(f"Error al iniciar el servidor: {e}")

if __name__ == "__main__":
    asyncio.run(main())