import random
import numpy as np


class Hamming74:
    """
    Implementation of Hamming(7,4) code.
    Encodes 4 bits of data into 7 bits.
    Can correct 1 bit error.
    """

    def __init__(self):
        # Generator matrix G (4x7)
        self.G = np.array(
            [
                [1, 1, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 1],
            ]
        )

        # Parity check matrix H (3x7)
        self.H = np.array(
            [[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1, 0], [0, 0, 1, 0, 1, 1, 1]]
        )

        # Syndrome to error position map
        # The syndrome is H * r^T. The result corresponds to a column in H.
        # We map the binary string of the syndrome to the 0-based index of the error.
        # Note: This mapping depends on the column order in H.
        self.syndrome_map = {
            (1, 0, 0): 0,
            (0, 1, 0): 1,
            (0, 0, 1): 2,
            (1, 1, 0): 3,
            (0, 1, 1): 4,
            (1, 1, 1): 5,
            (1, 0, 1): 6,
        }

    def encode(self, data_bits):
        """
        Encodes a list/array of bits. Length must be a multiple of 4.
        """
        data = np.array(data_bits)
        if len(data) % 4 != 0:
            raise ValueError("Data length must be a multiple of 4")

        n_blocks = len(data) // 4
        encoded_bits = []

        for i in range(n_blocks):
            block = data[i * 4 : (i + 1) * 4]
            # Matrix multiplication modulo 2
            encoded_block = np.dot(block, self.G) % 2
            encoded_bits.extend(encoded_block)

        return np.array(encoded_bits, dtype=int)

    def decode(self, received_bits):
        """
        Decodes a list/array of bits. Length must be a multiple of 7.
        Returns (decoded_bits, error_count)
        """
        received = np.array(received_bits)
        if len(received) % 7 != 0:
            raise ValueError("Received data length must be a multiple of 7")

        n_blocks = len(received) // 7
        decoded_bits = []
        corrected_errors = 0

        for i in range(n_blocks):
            block = received[i * 7 : (i + 1) * 7]

            # Calculate syndrome: z = H * r^T
            syndrome = np.dot(self.H, block) % 2
            syndrome_tuple = tuple(syndrome)

            if np.any(syndrome):
                # Error detected
                if syndrome_tuple in self.syndrome_map:
                    error_pos = self.syndrome_map[syndrome_tuple]
                    # Flip the bit to correct it
                    block[error_pos] = 1 - block[error_pos]
                    corrected_errors += 1

            # Extract data bits.
            # Based on G, the data bits are at indices 0, 1, 2, 3?
            # Let's check G structure.
            # G = [P | I_4] is standard systematic, but my G is not in that form.
            # My G rows are:
            # d1 -> 1101000
            # d2 -> 0110100
            # d3 -> 1110010
            # d4 -> 1010001
            # This G is not systematic (identity matrix is not clearly visible as a subblock).
            # Wait, actually looking at G:
            # Cols 3, 4, 5, 6 (0-indexed) seem to form identity?
            # Col 3: 1,0,0,0 (from d1) -> No
            # Let's look at the columns of G:
            # Col 0: 1,0,1,1
            # Col 1: 1,1,1,0
            # Col 2: 0,1,1,1
            # Col 3: 1,0,0,0 -> d1
            # Col 4: 0,1,0,0 -> d2
            # Col 5: 0,0,1,0 -> d3
            # Col 6: 0,0,0,1 -> d4
            # Yes! The data bits are at indices 3, 4, 5, 6.

            decoded_block = block[3:7]
            decoded_bits.extend(decoded_block)

        return np.array(decoded_bits, dtype=int), corrected_errors


class BinarySymmetricChannel:
    """
    Simulates a channel that flips bits with probability p.
    """

    def __init__(self, error_probability):
        self.p = error_probability

    def transmit(self, bits):
        """
        Passes bits through the noisy channel.
        """
        noise = np.random.choice([0, 1], size=len(bits), p=[1 - self.p, self.p])
        received_bits = (bits + noise) % 2
        return received_bits, np.sum(noise)


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
