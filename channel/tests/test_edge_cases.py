import numpy as np
import pytest
from channel_encoder import Hamming
from bsc import BinarySymmetricChannel


class TestEdgeCases:
    """
    Comprehensive edge case testing for Hamming encoder/decoder.
    """

    def test_minimum_data_single_bit(self):
        """Test with the absolute minimum: 1 bit."""
        hamming = Hamming(k=4, n=7)
        data = np.array([1])

        encoded = hamming.encode(data)
        decoded, errors = hamming.decode(encoded)

        assert len(decoded) == 1
        assert decoded[0] == 1
        assert errors == 0

    def test_minimum_data_zero_bit(self):
        """Test with a single zero bit."""
        hamming = Hamming(k=4, n=7)
        data = np.array([0])

        encoded = hamming.encode(data)
        decoded, errors = hamming.decode(encoded)

        assert len(decoded) == 1
        assert decoded[0] == 0
        assert errors == 0

    def test_exact_block_size_multiple_configs(self):
        """Test data sizes that exactly match block size for various configs."""
        configs = [
            (4, 7),
            (11, 15),
            (26, 31),
        ]

        for k, n in configs:
            hamming = Hamming(k=k, n=n)
            data = np.random.randint(0, 2, k)

            encoded = hamming.encode(data)
            assert len(encoded) == n

            decoded, errors = hamming.decode(encoded)
            assert np.array_equal(decoded, data)
            assert errors == 0

    def test_exact_multiple_blocks(self):
        """Test data sizes that are exact multiples of block size."""
        hamming = Hamming(k=4, n=7)

        for num_blocks in [1, 2, 5, 10, 100]:
            data = np.random.randint(0, 2, 4 * num_blocks)
            encoded = hamming.encode(data)

            assert len(encoded) == 7 * num_blocks

            decoded, errors = hamming.decode(encoded)
            assert np.array_equal(decoded, data)
            assert errors == 0

    def test_one_bit_less_than_block(self):
        """Test with k-1 bits (maximum padding scenario)."""
        configs = [
            (4, 7),
            (11, 15),
        ]

        for k, n in configs:
            hamming = Hamming(k=k, n=n)
            data = np.random.randint(0, 2, k - 1)

            encoded = hamming.encode(data)
            decoded, errors = hamming.decode(encoded)

            assert len(decoded) == k - 1
            assert np.array_equal(decoded, data)
            assert errors == 0

    def test_all_zeros_various_sizes(self):
        """Test all-zero data with various sizes."""
        hamming = Hamming(k=4, n=7)

        sizes = [1, 2, 3, 4, 5, 8, 15, 20, 100]
        for size in sizes:
            data = np.zeros(size, dtype=int)

            encoded = hamming.encode(data)
            decoded, errors = hamming.decode(encoded)

            assert len(decoded) == size
            assert np.array_equal(decoded, data)
            assert errors == 0

    def test_all_ones_various_sizes(self):
        """Test all-one data with various sizes."""
        hamming = Hamming(k=4, n=7)

        sizes = [1, 2, 3, 4, 5, 8, 15, 20, 100]
        for size in sizes:
            data = np.ones(size, dtype=int)

            encoded = hamming.encode(data)
            decoded, errors = hamming.decode(encoded)

            assert len(decoded) == size
            assert np.array_equal(decoded, data)
            assert errors == 0

    def test_alternating_pattern(self):
        """Test alternating 0-1 pattern."""
        hamming = Hamming(k=4, n=7)

        sizes = [10, 20, 50, 100]
        for size in sizes:
            data = np.array([i % 2 for i in range(size)])

            encoded = hamming.encode(data)
            decoded, errors = hamming.decode(encoded)

            assert len(decoded) == size
            assert np.array_equal(decoded, data)
            assert errors == 0

    def test_error_at_every_position_in_block(self):
        """Test error correction for errors at each position in a codeword."""
        hamming = Hamming(k=4, n=7)
        data = np.array([1, 0, 1, 1])

        for error_pos in range(7):
            encoded = hamming.encode(data)
            encoded[error_pos] = 1 - encoded[error_pos]

            decoded, errors = hamming.decode(encoded)

            assert np.array_equal(
                decoded, data
            ), f"Failed at error position {error_pos}"
            assert errors == 1

    def test_error_in_last_bit_of_block(self):
        """Test error correction when error is in the last bit of a block."""
        hamming = Hamming(k=4, n=7)
        data = np.random.randint(0, 2, 16)  # 4 blocks

        encoded = hamming.encode(data)

        # Introduce error in last bit of each block
        for block in range(4):
            encoded[block * 7 + 6] = 1 - encoded[block * 7 + 6]

        decoded, errors = hamming.decode(encoded)

        assert np.array_equal(decoded, data)
        assert errors == 4

    def test_error_in_first_bit_of_block(self):
        """Test error correction when error is in the first bit of a block."""
        hamming = Hamming(k=4, n=7)
        data = np.random.randint(0, 2, 16)  # 4 blocks

        encoded = hamming.encode(data)

        # Introduce error in first bit of each block
        for block in range(4):
            encoded[block * 7] = 1 - encoded[block * 7]

        decoded, errors = hamming.decode(encoded)

        assert np.array_equal(decoded, data)
        assert errors == 4

    def test_very_large_data(self):
        """Test with very large data size."""
        hamming = Hamming(k=4, n=7)
        data = np.random.randint(0, 2, 10000)

        encoded = hamming.encode(data)
        decoded, errors = hamming.decode(encoded)

        assert len(decoded) == 10000
        assert np.array_equal(decoded, data)
        assert errors == 0

    def test_random_data_sizes_no_padding(self):
        """Test random data sizes that don't require padding."""
        hamming = Hamming(k=4, n=7)

        for _ in range(20):
            num_blocks = np.random.randint(1, 50)
            data = np.random.randint(0, 2, 4 * num_blocks)

            encoded = hamming.encode(data)
            decoded, errors = hamming.decode(encoded)

            assert np.array_equal(decoded, data)
            assert errors == 0

    def test_random_data_sizes_with_padding(self):
        """Test random data sizes that require padding."""
        hamming = Hamming(k=4, n=7)

        for _ in range(20):
            size = np.random.randint(1, 200)
            data = np.random.randint(0, 2, size)

            encoded = hamming.encode(data)
            decoded, errors = hamming.decode(encoded)

            assert len(decoded) == size
            assert np.array_equal(decoded, data)
            assert errors == 0

    def test_100_percent_error_rate_small_data(self):
        """Test with 100% error rate on small data."""
        hamming = Hamming(k=4, n=7)
        channel = BinarySymmetricChannel(error_probability=1.0)

        data = np.array([1, 0, 1, 1])
        encoded = hamming.encode(data)
        received, _ = channel.transmit(encoded)

        # With 100% error rate, all bits are flipped
        # Hamming can only correct 1 error per block, so this should fail
        decoded, _ = hamming.decode(received)

        # We expect the decoding to not match (since all bits flipped)
        # This tests that the decoder doesn't crash with extreme inputs
        assert len(decoded) == len(data)

    def test_sequential_encoding_decoding(self):
        """Test encoding and decoding the same data multiple times."""
        hamming = Hamming(k=4, n=7)
        data = np.random.randint(0, 2, 20)

        for _ in range(10):
            encoded = hamming.encode(data)
            decoded, errors = hamming.decode(encoded)

            assert np.array_equal(decoded, data)
            assert errors == 0

    def test_different_instances_same_config(self):
        """Test that different Hamming instances with same config work identically."""
        data = np.random.randint(0, 2, 20)

        hamming1 = Hamming(k=4, n=7)
        hamming2 = Hamming(k=4, n=7)

        encoded1 = hamming1.encode(data)
        encoded2 = hamming2.encode(data)

        # Both should produce the same encoding
        assert np.array_equal(encoded1, encoded2)

        # Both should decode correctly
        decoded1, _ = hamming1.decode(encoded1)
        decoded2, _ = hamming2.decode(encoded2)

        assert np.array_equal(decoded1, data)
        assert np.array_equal(decoded2, data)


def run_edge_case_tests():
    """Run all edge case tests and print results."""
    print("\n" + "=" * 70)
    print("  RUNNING EDGE CASE TESTS")
    print("=" * 70 + "\n")

    pytest.main([__file__, "-v", "--tb=short", "-k", "TestEdgeCases"])


if __name__ == "__main__":
    run_edge_case_tests()
