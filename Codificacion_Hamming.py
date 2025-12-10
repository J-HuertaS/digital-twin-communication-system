import random
import numpy as np

class Hamming:
    def __init__(self, k: int = 4, n: int = 7):
        self.n = n
        self.k = k
        self._check_valid_parameters(n, k)
        self.m = n - k

        self.H, self.P = self._build_H()
    
        self.G = self._build_G()

        self._check_valid_G_H()

        self.syndrome_map = self._build_syndrome_map()
        self.data_length = 0

    # Verifica que n > k
    def _check_valid_parameters(self, n, k):
        assert n > k

    def _check_valid_G_H(self):
        # Verifica que las matrices G y H sean válidas para el código Hamming(n,k).
        product = np.dot(self.G, self.H.T) % 2
        assert np.all(product == 0)

    def _get_non_zero_vectors(self, I_m: np.ndarray):
        """
        Parametros:
        I_m (np.ndarray): Matriz identidad de tamaño m x m.

        Retorna:
        all_vectors: Lista de vectores columna binarios de longitud m.
        """
        all_vectors = []
        for i in range(1, 2**self.m):
            # Convertir el número (i) a su representación binaria de m bits
            vector = []
            temp = i
            for _ in range(self.m):
                vector.insert(0, temp % 2)
                temp //= 2
            if not vector in I_m.tolist():
                all_vectors.append(vector)
        return np.array(all_vectors)

    def _build_H(self):
        """
        Construye la matriz de paridad H para el código Hamming(n,k).
         Retorna:
        H (np.ndarray): Matriz de paridad de tamaño mxn.
        P (np.ndarray): Matriz de paridad de tamaño kxm.
        """
        I_m = np.eye(self.m, dtype=int)
        parity_matrix = self._get_non_zero_vectors(I_m)
        H = np.hstack((parity_matrix.T,I_m))
        return H,parity_matrix

    def _build_G(self):
        """"
        Construye la matriz generadora G para el código Hamming(n,k).
        Retorna:
        G (np.ndarray): Matriz generadora de tamaño kxn."""
        return np.hstack((np.eye(self.k, dtype=int), self.P))

    def _build_syndrome_map(self):
        """
        Construye el mapa de síndromes a posiciones de error.
        Retorna:
        syndrome_map (dict): Mapa de síndromes a posiciones de error.
        """
        syndrome_map = {}
        for col in range(self.n):
            syndrome = tuple(self.H[:, col])
            syndrome_map[syndrome] = col
        return syndrome_map
    
    def _apply_padding(self, data_bits):
        """
        Aplica padding a los bits de datos para que su longitud sea múltiplo de k.
        Parametros:
        data_bits (list or np.ndarray): Lista o array de bits de datos (0s y 1s).
        Retorna:
        padded_data (np.ndarray): Bits de datos con padding aplicado.
        """
        data = np.array(data_bits)
        remainder = len(data) % self.k
        if remainder != 0:
            padding_length = self.k - remainder
            padding = np.zeros(padding_length, dtype=int)
            padded_data = np.concatenate((data, padding))
            return padded_data
        return data
    
    def encode(self, data_bits):
        """
        Codifica un array de strings binarios que representan bits de datos.
        Cálcula c = d * G 
        """
        self.data_length = len(data_bits)
        data = self._apply_padding(data_bits)

        n_blocks = len(data) // self.k
        encoded_bits = []

        for i in range(n_blocks):
            block = data[i * self.k : (i + 1) * self.k]
            # Matrix multiplication modulo 2
            encoded_block = np.dot(block, self.G) % 2
            encoded_bits.extend(encoded_block)
        return np.array(encoded_bits, dtype=int)

    def decode(self, received_bits):
        """
        Descodifica y corrige un array de strings binarios que representan bits recibidos.
        Cálcula el síndrome z = H * r^T y corrige errores si es necesario.
        """
        received = np.array(received_bits)
        if len(received) % self.n != 0:
            raise ValueError(f"Received data length must be a multiple of {self.n}")

        n_blocks = len(received) // self.n
        decoded_bits = []
        corrected_errors = 0

        for i in range(n_blocks):
            block = received[i * self.n : (i + 1) * self.n]

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

            decoded_block = block[0:self.k]
            decoded_bits.extend(decoded_block)
        # Deshacer el padding
        restored_data = decoded_bits[: self.data_length]
        return np.array(restored_data, dtype=int), corrected_errors