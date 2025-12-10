import numpy as np
from math import log2

ADC_RESOLUTION_BITS = 10
ADC_MAX_VALUE = 2**ADC_RESOLUTION_BITS - 1
REFERENCE_VOLTAGE = 5.0

def apply_moving_average_filter(data_array, window):
    """Aplicación de filtro de promedio móvil."""
    weights = np.ones(window) / window
    return np.convolve(data_array, weights, mode='valid')

def adc_to_voltage(raw_array):
    """Convierte datos del ADC (0–1023) a voltaje."""
    return (raw_array / ADC_MAX_VALUE) * REFERENCE_VOLTAGE

def voltage_to_adc(voltage_array):
    """Convierte voltaje filtrado a valores ADC."""
    normalized = (voltage_array / REFERENCE_VOLTAGE) * ADC_MAX_VALUE
    return np.round(normalized).astype(int)

def calculate_entropy(message_array):
    """
    Calcula la entropía Shannon de un conjunto de símbolos.
    El mensaje debe ser un array de enteros (ej: datos ADC filtrados).
    """
    if len(message_array) == 0:
        return 0.0

    # Conteo de ocurrencias
    values, counts = np.unique(message_array, return_counts=True)

    # Probabilidades
    probabilities = counts / len(message_array)

    # Entropía Shannon
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy
