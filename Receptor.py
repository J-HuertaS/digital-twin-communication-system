# receptor.py
import asyncio
import logging
import json
import base64
from typing import List

import numpy as np
import websockets

from Codificacion_Hamming import Hamming
from Codificacion_Huffman import decode as huffman_decode

# CONFIG
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8765

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("receptor")

def packed_bytes_to_bits_list(b: bytes, padding: int) -> List[int]:
    if not b:
        return []
    bits = "".join(f"{byte:08b}" for byte in b)
    if padding:
        bits = bits[:-padding]
    return [int(ch) for ch in bits]

async def handle_connection(websocket, path):
    logger.info(f"Conexión entrante: {websocket.remote_address}")
    try:
        async for message in websocket:
            logger.info("Mensaje recibido por websocket (raw).")
            try:
                payload = json.loads(message)
            except Exception as e:
                logger.error(f"JSON inválido: {e}")
                await websocket.send("ERR: invalid json")
                continue

            # Validar tipo
            if payload.get("type") != "block":
                logger.warning("Mensaje desconocido (no 'block'). Ignorando.")
                await websocket.send("IGNORED")
                continue

            logger.info(f"--- Procesando bloque recibido ---")
            logger.info(f"Bloque original (reportado por emisor): {payload.get('original_block')}")
            logger.info(f"Entropy (reportado): {payload.get('entropy')}")

            # Obtener parámetros y datos
            codebook = payload["huffman_codebook"]
            hamming_k = payload["hamming_k"]
            hamming_n = payload["hamming_n"]
            hamming_data_length = payload["hamming_data_length"]
            hamming_b64 = payload["hamming_bits_base64"]
            hamming_padding = payload["hamming_bytes_padding"]

            # Decodificar base64 -> bytes -> lista de bits
            try:
                hb = base64.b64decode(hamming_b64.encode())
            except Exception as e:
                logger.error(f"Error decodificando base64: {e}")
                await websocket.send("ERR: base64")
                continue
            hamming_bits_list = packed_bytes_to_bits_list(hb, hamming_padding)
            logger.info(f"Bits hamming recibidos (desempaquetados) len={len(hamming_bits_list)}")

            # Crear instancia Hamming igual al emisor
            h = Hamming(k=hamming_k, n=hamming_n)
            # IMPORTANTE: setear data_length al valor transmitido por el emisor
            h.data_length = hamming_data_length

            # Decodificar y corregir errores (si existieran)
            try:
                decoded_bits_array, corrected = h.decode(hamming_bits_list)
                decoded_bits_list = decoded_bits_array.tolist()
            except Exception as e:
                logger.exception(f"Error decodificando Hamming: {e}")
                await websocket.send("ERR: hamming decode")
                continue

            logger.info(f"Hamming decode: errores corregidos = {corrected}")
            logger.info(f"Bits Huffman restaurados (len={len(decoded_bits_list)}): {decoded_bits_list[:160]}{'...' if len(decoded_bits_list)>160 else ''}")

            # Convertir bits a string y usar Huffman decode
            bits_string = "".join(str(b) for b in decoded_bits_list)
            logger.info(f"Bits string para Huffman: {bits_string[:160]}{'...' if len(bits_string)>160 else ''}")

            try:
                decoded_symbols = huffman_decode(bits_string, codebook)
            except Exception as e:
                logger.exception(f"Error decodificando Huffman: {e}")
                await websocket.send("ERR: huffman decode")
                continue

            logger.info(f"Mensaje final decodificado (símbolos): {decoded_symbols}")
            # opcional: devolver ACK con resumen
            ack = {
                "status": "ok",
                "corrected_errors": corrected,
                "decoded_length": len(decoded_symbols)
            }
            await websocket.send(json.dumps(ack))

    except websockets.ConnectionClosed:
        logger.info("Conexión cerrada por el cliente.")
    except Exception as e:
        logger.exception(f"Error en handle_connection: {e}")

async def main():
    server = await websockets.serve(handle_connection, LISTEN_HOST, LISTEN_PORT)
    logger.info(f"Servidor WebSocket escuchando en ws://{LISTEN_HOST}:{LISTEN_PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Receptor detenido por usuario.")
