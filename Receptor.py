import asyncio
import json
import logging
import websockets
import numpy as np

# Importación de módulos de codificación y filtrado
from Codificacion_Hamming import Hamming
from Codificacion_Huffman import decode as huf_decode
from Filtrado import calculate_entropy, voltage_to_adc, adc_to_voltage, ADC_RESOLUTION_BITS

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Parámetros del sistema
# Hamming(7,4) es usado por el Emisor
HAMMING_K = 4
HAMMING_N = 7


async def receive_message():
    """Se conecta al servidor EMISOR, recibe los datos codificados y los decodifica."""

    uri = "ws://localhost:8765"
    
    # Inicializar la instancia de Hamming para la decodificación
    hamming = Hamming(k=HAMMING_K, n=HAMMING_N)

    try:
        logging.info(f"Intentando conectar a {uri}...")
        async with websockets.connect(uri) as websocket:
            logging.info("Conexión establecida con el EMISOR.")

            while True:
                # ------------------------------
                # 1. RECEPCIÓN DE DATOS
                # ------------------------------
                
                message_json = await websocket.recv()
                message = json.loads(message_json)
                
                # Extraer componentes del mensaje
                entropy_sent = message.get("entropy")
                codebook = message.get("codebook")
                hamming_encoded_list = message.get("hamming")

                if not codebook or not hamming_encoded_list:
                    logging.warning("Mensaje incompleto o inválido recibido. Ignorando.")
                    continue

                logging.info(f"Mensaje recibido. Entropía reportada: {entropy_sent:.4f}")
                logging.info(f"Tamaño de datos Hamming: {len(hamming_encoded_list)} bits")
                logging.info(f"Diccionario de Huffman recibido: {codebook}")
                
                # Convertir los bits de Hamming de lista a array numpy
                received_hamming_bits = np.array(hamming_encoded_list, dtype=int)
                
                # ------------------------------
                # 2. DECODIFICACIÓN Y CORRECCIÓN HAMMING
                # ------------------------------
                
                logging.info("Iniciando decodificación Hamming (Corrección de Errores)...")
                
                # Decodificar Hamming: devuelve los bits de Huffman y el número de errores corregidos
                huffman_bits_corrected_array, corrected_errors = hamming.decode(
                    received_hamming_bits, data_length=message.get("huffman_length")
                )
                
                # Convertir a cadena de bits para Huffman
                huffman_bits_corrected_str = "".join(map(str, huffman_bits_corrected_array.tolist()))

                logging.info(f"Errores corregidos por Hamming: {corrected_errors}")
                logging.info(f"Bits Huffman corregidos (longitud {len(huffman_bits_corrected_str)}): {huffman_bits_corrected_str}...")

                # ------------------------------
                # 3. DECODIFICACIÓN HUFFMAN
                # ------------------------------
                
                logging.info("Iniciando decodificación Huffman (Descompresión)...")
                
                # Se necesita invertir el diccionario para que las claves sean strings binarias
                # La función huf_decode del módulo Codificacion_Huffman.py ya maneja la inversión.
                
                # Los símbolos decodificados serán enteros (valores ADC discretos)
                try:
                    # El codebook recibido tiene claves de string (ej. '400') debido a json.dumps.
                    # Debemos convertir las claves de vuelta a enteros antes de pasarlas
                    # a la función huf_decode, ya que el EMISOR usó enteros como símbolos.
                    codebook_int = {int(k): v for k, v in codebook.items()}
                    
                    decoded_adc_values = huf_decode(huffman_bits_corrected_str, codebook_int)
                    
                    # Convertir la lista de símbolos (enteros) a un array de numpy
                    recovered_adc_array = np.array(decoded_adc_values, dtype=int)
                    
                except ValueError as e:
                    logging.error(f"Error durante la decodificación Huffman: {e}. Puede ser un error de canal no corregido.")
                    continue
                
                # ------------------------------
                # 4. RESULTADOS Y ANÁLISIS
                # ------------------------------
                
                logging.info(f"Datos ADC recuperados ({len(recovered_adc_array)} samples): {recovered_adc_array.tolist()}")
                
                # Calcular la entropía de los datos recuperados para comparación
                entropy_recovered = calculate_entropy(recovered_adc_array)
                
                # Mostrar el resultado final
                logging.info("--- Resumen del Bloque ---")
                logging.info(f"Entropía reportada (Emisor): {entropy_sent:.4f}")
                logging.info(f"Entropía de datos recuperados: {entropy_recovered:.4f}")
                logging.info(f"Número de errores corregidos por Hamming: {corrected_errors}")
                logging.info("--------------------------\n")
                
    except websockets.exceptions.ConnectionClosedOK:
        logging.info("Conexión cerrada por el EMISOR.")
    except ConnectionRefusedError:
        logging.error("No se pudo conectar. Asegúrate de que el servidor EMISOR esté en ejecución en ws://localhost:8765.")
    except Exception as e:
        logging.error(f"Error inesperado en RECEPTOR: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(receive_message())