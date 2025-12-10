from collections import Counter
import heapq
from typing import Any, Dict, List, Tuple


class _Node:
    # Nodo interno para el árbol de Huffman.
    def __init__(self, symbol=None, freq: int = 0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other: "_Node"):
        return self.freq < other.freq


def build_huffman_tree(freqs: Dict[Any, int]) -> _Node:
    heap = [_Node(sym, freq) for sym, freq in freqs.items()]

    if len(heap) == 1:
        node = heap[0]
        return _Node(None, node.freq, node, None)

    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = _Node(None, n1.freq + n2.freq, n1, n2)
        heapq.heappush(heap, merged)

    return heap[0]


def build_codebook(root: _Node) -> Dict[Any, str]:
    # Genera el diccionario a partir del codebook
    codebook: Dict[Any, str] = {}

    def traverse(node: _Node, prefix: str):
        if node.symbol is not None:
            codebook[node.symbol] = prefix or "0"
            return
        traverse(node.left, prefix + "0")
        if node.right is not None:
            traverse(node.right, prefix + "1")

    traverse(root, "")
    return codebook



def train_codebook(data: List[Any]) -> Dict[Any, str]:
    # Entrena el codebook de Huffman a partir de la data
    freqs = Counter(data)
    if not freqs:
        raise ValueError("No se puede entrenar Huffman con data vacía.")
    root = build_huffman_tree(freqs)
    return build_codebook(root)


def encode(data: List[Any], codebook: Dict[Any, str]) -> str:
    # Codifica la secuencia con el codebook
    try:
        return "".join(codebook[sym] for sym in data)
    except KeyError as e:
        raise ValueError(
            f"Símbolo {e.args[0]!r} no está en el codebook"
        )


def decode(bits: str, codebook: Dict[Any, str]) -> List[Any]:
    # Decodifica la secuencia con el codebook
    rev = {code: sym for sym, code in codebook.items()}

    decoded: List[Any] = []
    current = ""

    for b in bits:
        current += b
        if current in rev:
            decoded.append(rev[current])
            current = ""

    if current:
        raise ValueError("Error de canal, no es un codigo valido")

    return decoded


def bits_to_bytes(bits: str) -> Tuple[bytes, int]:
    # Convierte bits a bytes para una transmisión eficiente
    if not bits:
        return b"", 0

    padding = (8 - (len(bits) % 8)) % 8
    bits_padded = bits + "0" * padding

    byte_array = bytearray()
    for i in range(0, len(bits_padded), 8):
        byte_chunk = bits_padded[i:i + 8]
        byte_array.append(int(byte_chunk, 2))

    return bytes(byte_array), padding


def bytes_to_bits(data: bytes, padding: int) -> str:
    # Convierte de bytes a bits 

    bits = "".join(f"{byte:08b}" for byte in data)
    if padding:
        bits = bits[:-padding]
    return bits