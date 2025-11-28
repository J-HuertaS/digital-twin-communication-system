import numpy as np
from channel_encoder import Hamming74
from bsc import BinarySymmetricChannel

def generate_random_data(num_bits):
    return np.random.randint(0, 2, num_bits)


def bits_to_string(bits):
    return "".join(map(str, bits))


def run_simulation():
    print(
        "=== Simulación de Sistema de Comunicación (Canal + Corrección de Errores) ==="
    )

    # Configuración
    NUM_DATA_BITS = 20  # Múltiplo de 4 para Hamming(7,4)
    ERROR_PROBABILITY = 0.1  # 10% de probabilidad de error por bit

    print(
        f"Configuración: {NUM_DATA_BITS} bits de datos, Probabilidad de error del canal: {ERROR_PROBABILITY}"
    )
    print("-" * 60)

    # 1. Fuente
    source_data = generate_random_data(NUM_DATA_BITS)
    print(f"1. Datos de la Fuente (Original): \n   {bits_to_string(source_data)}")

    # 2. Codificación de Canal
    hamming = Hamming74()
    encoded_data = hamming.encode(source_data)
    print(f"\n2. Datos Codificados (Hamming 7,4): \n   {bits_to_string(encoded_data)}")
    print(
        f"   (Longitud original: {len(source_data)} -> Longitud codificada: {len(encoded_data)})"
    )

    # 3. Canal Ruidoso
    channel = BinarySymmetricChannel(ERROR_PROBABILITY)
    received_data, errors_introduced = channel.transmit(encoded_data)

    # Visualizar errores
    diff_str = ""
    for s, r in zip(encoded_data, received_data):
        if s != r:
            diff_str += "^"  # Marca el error
        else:
            diff_str += " "

    print(
        f"\n3. Datos Recibidos del Canal (con ruido): \n   {bits_to_string(received_data)}"
    )
    print(f"   {diff_str}")
    print(f"   Errores introducidos por el canal: {errors_introduced}")

    # 4. Decodificación y Corrección
    decoded_data, errors_corrected = hamming.decode(received_data)
    print(f"\n4. Datos Decodificados y Corregidos: \n   {bits_to_string(decoded_data)}")
    print(f"   Errores corregidos por el algoritmo: {errors_corrected}")

    # 5. Verificación Final
    bit_errors_final = np.sum(source_data != decoded_data)
    print("-" * 60)
    print(f"Resumen:")
    print(f"Errores totales introducidos: {errors_introduced}")
    print(f"Errores corregidos: {errors_corrected}")
    print(f"Errores finales en el mensaje: {bit_errors_final}")

    if bit_errors_final == 0:
        print(
            "\n¡ÉXITO! La transmisión fue perfecta gracias a la corrección de errores."
        )
    else:
        print(
            "\nADVERTENCIA: Hubo errores que no se pudieron corregir (probablemente más de 1 error por bloque)."
        )


if __name__ == "__main__":
    run_simulation()