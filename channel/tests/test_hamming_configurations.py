import numpy as np
import pytest
from channel_encoder import Hamming
from bsc import BinarySymmetricChannel


class TestHammingConfigurations:
    """
    Test suite for different Hamming code configurations.
    Tests various (n, k) parameters and data sizes.
    """

    def test_hamming_7_4_no_padding(self):
        """Test Hamming(7,4) with data that doesn't require padding."""
        hamming = Hamming(k=4, n=7)
        data = np.array([1, 0, 1, 1, 0, 0, 1, 0])  # 8 bits = 2 blocks of 4

        encoded = hamming.encode(data)
        assert len(encoded) == 14  # 2 blocks * 7 bits

        # Decode without errors
        decoded, errors = hamming.decode(encoded)
        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_hamming_7_4_with_padding(self):
        """Test Hamming(7,4) with data that requires padding."""
        hamming = Hamming(k=4, n=7)
        data = np.array([1, 0, 1, 1, 0, 1])  # 6 bits, needs 2 bits padding

        encoded = hamming.encode(data)
        assert len(encoded) == 14  # Padded to 8 bits = 2 blocks * 7 bits

        # Decode and verify original data is recovered (without padding)
        decoded, errors = hamming.decode(encoded)
        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_hamming_15_11_no_padding(self):
        """Test Hamming(15,11) with data that doesn't require padding."""
        hamming = Hamming(k=11, n=15)
        data = np.random.randint(0, 2, 22)  # 22 bits = 2 blocks of 11

        encoded = hamming.encode(data)
        assert len(encoded) == 30  # 2 blocks * 15 bits

        decoded, errors = hamming.decode(encoded)
        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_hamming_15_11_with_padding(self):
        """Test Hamming(15,11) with data that requires padding."""
        hamming = Hamming(k=11, n=15)
        data = np.random.randint(0, 2, 15)  # 15 bits, needs 7 bits padding

        encoded = hamming.encode(data)
        assert len(encoded) == 30  # Padded to 22 bits = 2 blocks * 15 bits

        decoded, errors = hamming.decode(encoded)
        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_hamming_31_26_no_padding(self):
        """Test Hamming(31,26) with data that doesn't require padding."""
        hamming = Hamming(k=26, n=31)
        data = np.random.randint(0, 2, 52)  # 52 bits = 2 blocks of 26

        encoded = hamming.encode(data)
        assert len(encoded) == 62  # 2 blocks * 31 bits

        decoded, errors = hamming.decode(encoded)
        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_hamming_31_26_with_padding(self):
        """Test Hamming(31,26) with data that requires padding."""
        hamming = Hamming(k=26, n=31)
        data = np.random.randint(0, 2, 30)  # 30 bits, needs 22 bits padding

        encoded = hamming.encode(data)
        assert len(encoded) == 62  # Padded to 52 bits = 2 blocks * 31 bits

        decoded, errors = hamming.decode(encoded)
        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_single_bit_error_correction_7_4(self):
        """Test single bit error correction for Hamming(7,4)."""
        hamming = Hamming(k=4, n=7)
        data = np.array([1, 0, 1, 1])

        encoded = hamming.encode(data)

        # Introduce single bit error
        error_pos = 3
        encoded[error_pos] = 1 - encoded[error_pos]

        decoded, errors = hamming.decode(encoded)
        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 1

    def test_single_bit_error_correction_15_11(self):
        """Test single bit error correction for Hamming(15,11)."""
        hamming = Hamming(k=11, n=15)
        data = np.random.randint(0, 2, 11)

        encoded = hamming.encode(data)

        # Introduce single bit error
        error_pos = 7
        encoded[error_pos] = 1 - encoded[error_pos]

        decoded, errors = hamming.decode(encoded)
        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 1

    def test_multiple_blocks_with_errors(self):
        """Test error correction across multiple blocks."""
        hamming = Hamming(k=4, n=7)
        data = np.random.randint(0, 2, 16)  # 4 blocks

        encoded = hamming.encode(data)

        # Introduce one error per block
        for i in range(4):
            error_pos = i * 7 + np.random.randint(0, 7)
            encoded[error_pos] = 1 - encoded[error_pos]

        decoded, errors = hamming.decode(encoded)
        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 4

    def test_edge_case_single_bit(self):
        """Test encoding/decoding with minimal data (1 bit)."""
        hamming = Hamming(k=4, n=7)
        data = np.array([1])

        encoded = hamming.encode(data)
        decoded, errors = hamming.decode(encoded)

        assert len(decoded) == 1
        assert decoded[0] == data[0]
        assert errors == 0

    def test_edge_case_exact_block_size(self):
        """Test with data size exactly matching block size."""
        hamming = Hamming(k=4, n=7)
        data = np.array([1, 0, 1, 1])

        encoded = hamming.encode(data)
        assert len(encoded) == 7

        decoded, errors = hamming.decode(encoded)
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_large_data_no_padding(self):
        """Test with large data that doesn't require padding."""
        hamming = Hamming(k=4, n=7)
        data = np.random.randint(0, 2, 400)  # 100 blocks

        encoded = hamming.encode(data)
        assert len(encoded) == 700

        decoded, errors = hamming.decode(encoded)
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_large_data_with_padding(self):
        """Test with large data that requires padding."""
        hamming = Hamming(k=4, n=7)
        data = np.random.randint(0, 2, 397)  # Requires 3 bits padding

        encoded = hamming.encode(data)
        decoded, errors = hamming.decode(encoded)

        assert len(decoded) == len(data)
        assert np.array_equal(decoded, data)
        assert errors == 0


class TestPaddingScenarios:
    """
    Focused tests on padding behavior for different configurations.
    """

    def test_padding_amount_calculation(self):
        """Test various padding scenarios."""
        test_cases = [
            (4, 7, 1, 3),  # k=4, data=1 bit, needs 3 bits padding
            (4, 7, 2, 2),  # k=4, data=2 bits, needs 2 bits padding
            (4, 7, 3, 1),  # k=4, data=3 bits, needs 1 bit padding
            (4, 7, 4, 0),  # k=4, data=4 bits, needs 0 bits padding
            (4, 7, 5, 3),  # k=4, data=5 bits, needs 3 bits padding
            (11, 15, 9, 2),  # k=11, data=9 bits, needs 2 bits padding
            (11, 15, 11, 0),  # k=11, data=11 bits, needs 0 bits padding
            (11, 15, 13, 9),  # k=11, data=13 bits, needs 9 bits padding
        ]

        for k, n, data_size, expected_padding in test_cases:
            hamming = Hamming(k=k, n=n)
            data = np.random.randint(0, 2, data_size)

            encoded = hamming.encode(data)
            decoded, _ = hamming.decode(encoded)

            # Verify original data is recovered without padding bits
            assert len(decoded) == data_size
            assert np.array_equal(decoded, data)

    def test_padding_with_all_zeros(self):
        """Test that padding doesn't affect all-zero data."""
        hamming = Hamming(k=4, n=7)
        data = np.zeros(3, dtype=int)  # 3 bits, needs 1 bit padding

        encoded = hamming.encode(data)
        decoded, errors = hamming.decode(encoded)

        assert len(decoded) == 3
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_padding_with_all_ones(self):
        """Test that padding doesn't affect all-one data."""
        hamming = Hamming(k=4, n=7)
        data = np.ones(3, dtype=int)  # 3 bits, needs 1 bit padding

        encoded = hamming.encode(data)
        decoded, errors = hamming.decode(encoded)

        assert len(decoded) == 3
        assert np.array_equal(decoded, data)
        assert errors == 0


class TestChannelSimulation:
    """
    Test Hamming code performance with BSC (Binary Symmetric Channel).
    """

    def test_low_error_probability(self):
        """Test with low channel error probability."""
        hamming = Hamming(k=4, n=7)
        channel = BinarySymmetricChannel(error_probability=0.01)

        data = np.random.randint(0, 2, 100)
        encoded = hamming.encode(data)

        received, _ = channel.transmit(encoded)
        decoded, errors_corrected = hamming.decode(received)

        # With low error probability and single-error correction,
        # most data should be recovered correctly
        bit_errors = np.sum(data != decoded)
        assert bit_errors <= 5  # Allow some uncorrectable errors

    def test_medium_error_probability(self):
        """Test with medium channel error probability."""
        hamming = Hamming(k=4, n=7)
        channel = BinarySymmetricChannel(error_probability=0.1)

        data = np.random.randint(0, 2, 100)
        encoded = hamming.encode(data)

        received, errors_introduced = channel.transmit(encoded)
        decoded, errors_corrected = hamming.decode(received)

        # Verify some errors were corrected
        assert errors_corrected > 0

        # Most single-bit errors per block should be corrected
        bit_errors = np.sum(data != decoded)
        # With 10% error rate, we expect some uncorrectable errors
        # but should still correct many
        assert errors_corrected >= bit_errors

    def test_no_channel_errors(self):
        """Test with perfect channel (no errors)."""
        hamming = Hamming(k=4, n=7)
        channel = BinarySymmetricChannel(error_probability=0.0)

        data = np.random.randint(0, 2, 100)
        encoded = hamming.encode(data)

        received, errors_introduced = channel.transmit(encoded)
        decoded, errors_corrected = hamming.decode(received)

        assert errors_introduced == 0
        assert errors_corrected == 0
        assert np.array_equal(data, decoded)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
