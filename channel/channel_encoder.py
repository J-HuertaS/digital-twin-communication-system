import random
import numpy as np


class Hamming74:
    """
    Implementation of Hamming(7,4) code.
    Encodes 4 bits of data into 7 bits.
    Can correct 1 bit error.
    """

    def __init__(self, k: int = 4, n: int = 7):
        self.n = n
        self.k = k
        self._check_valid_parameters(n, k)
        self.m = n - k
        
        self.H, self.P = self._build_H()
        print(self.H,self.H.shape)
    
        self.G = self._build_G()
        print(self.G,self.G.shape)

        self._check_valid_G_H()

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

    def _check_valid_parameters(self, n, k):
        assert n > k, "n debe ser mayor a k"

    def _check_valid_G_H(self):
        """"
        Verifica que las matrices G y H sean válidas para el código Hamming(n,k).
        """
        # Verificar G * H^T = 0
        product = np.dot(self.G, self.H.T) % 2
        assert np.all(product == 0), "G * H^T is not zero matrix"

    def get_non_zero_vectors(self, I_m: np.ndarray):
        """
        Genera todos los vectores columna binarios de longitud m, excepto el vector cero y los pertenecientes a la matriz identidad de tamaño m.
        
        Parametros:
        I_m (np.ndarray): Matriz identidad de tamaño m x m.

        Retorna:
        all_vectors: Lista de vectores columna binarios de longitud m.
        """
        all_vectors = []
        # Iterar desde 1 hasta 2^m - 1
        for i in range(1, 2**self.m):
            # Convertir el número (i) a su representación binaria de m bits
            vector = []
            temp = i
            for _ in range(self.m):
                # Obtener el bit menos significativo
                vector.insert(0, temp % 2)
                temp //= 2
            if not vector in I_m.tolist():
                all_vectors.append(vector)
        # Retorna una lista de listas (vectores)
        return np.array(all_vectors)

    def _build_H(self):
        """
        Construye la matriz de paridad H para el código Hamming(n,k).
         Retorna:
        H (np.ndarray): Matriz de paridad de tamaño mxn.
        P (np.ndarray): Matriz de paridad de tamaño kxm.
        """
        I_m = np.eye(self.m, dtype=int)
        parity_matrix = self.get_non_zero_vectors(I_m)
        H = np.hstack((parity_matrix.T,I_m))
        return H,parity_matrix

    def _build_G(self):
        """"
        Construye la matriz generadora G para el código Hamming(n,k).
        Retorna:
        G (np.ndarray): Matriz generadora de tamaño kxn."""
        return np.hstack((np.eye(self.k, dtype=int), self.P))

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
