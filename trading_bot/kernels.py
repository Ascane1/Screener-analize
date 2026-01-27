import numpy as np
from enum import Enum
from typing import Callable


class KernelType(Enum):
    TRIANGULAR = "Triangular"
    GAUSSIAN = "Gaussian"
    EPANECHNIKOV = "Epanechnikov"
    LOGISTIC = "Logistic"
    LOG_LOGISTIC = "Log Logistic"
    COSINE = "Cosine"
    SINC = "Sinc"
    LAPLACE = "Laplace"
    QUARTIC = "Quartic"
    PARABOLIC = "Parabolic"
    EXPONENTIAL = "Exponential"
    SILVERMAN = "Silverman"
    CAUCHY = "Cauchy"
    TENT = "Tent"
    WAVE = "Wave"
    POWER = "Power"
    MORTERS = "Morters"


def gaussian(x: float, bw: float) -> float:
    return np.exp(-(x / bw) ** 2 / 2) / np.sqrt(2 * np.pi)


def triangular(x: float, bw: float) -> float:
    return 1 - abs(x / bw) if abs(x / bw) <= 1 else 0.0


def epanechnikov(x: float, bw: float) -> float:
    return 0.75 * (1 - (x / bw) ** 2) if abs(x / bw) <= 1 else 0.0


def quartic(x: float, bw: float) -> float:
    return (15 / 16) * (1 - (x / bw) ** 2) ** 2 if abs(x / bw) <= 1 else 0.0


def logistic(x: float, bw: float) -> float:
    return 1 / (np.exp(x / bw) + 2 + np.exp(-x / bw))


def log_logistic(x: float, bw: float) -> float:
    return 1 / (1 + abs(x / bw)) ** 2


def cosine(x: float, bw: float) -> float:
    if abs(x / bw) <= 1:
        return (np.pi / 4) * np.cos((np.pi / 2) * (x / bw))
    return 0.0


def sinc(x: float, bw: float) -> float:
    if x == 0:
        return 1.0
    return np.sin(np.pi * x / bw) / (np.pi * x / bw)


def laplace(x: float, bw: float) -> float:
    return (1 / (2 * bw)) * np.exp(-abs(x / bw))


def exponential(x: float, bw: float) -> float:
    return (1 / bw) * np.exp(-abs(x / bw))


def silverman(x: float, bw: float) -> float:
    if abs(x / bw) <= 0.5:
        return 0.5 * np.exp(-(x / bw) / 2) * np.sin((x / bw) / 2 + np.pi / 4)
    return 0.0


def tent(x: float, bw: float) -> float:
    return 1 - abs(x / bw) if abs(x / bw) <= 1 else 0.0


def cauchy(x: float, bw: float) -> float:
    return 1 / (np.pi * bw * (1 + (x / bw) ** 2))


def wave(x: float, bw: float) -> float:
    if abs(x / bw) <= 1:
        return (1 - abs(x / bw)) * np.cos((np.pi * x) / bw)
    return 0.0


def parabolic(x: float, bw: float) -> float:
    return 1 - (x / bw) ** 2 if abs(x / bw) <= 1 else 0.0


def power(x: float, bw: float) -> float:
    if abs(x / bw) <= 1:
        return (1 - abs(x / bw) ** 3) ** 3
    return 0.0


def morters(x: float, bw: float) -> float:
    if abs(x / bw) <= np.pi:
        return (1 + np.cos(x / bw)) / (2 * np.pi * bw)
    return 0.0


def get_kernel_function(kernel_type: KernelType) -> Callable:
    """Возвращает функцию ядра по типу"""
    kernel_map = {
        KernelType.TRIANGULAR: triangular,
        KernelType.GAUSSIAN: gaussian,
        KernelType.EPANECHNIKOV: epanechnikov,
        KernelType.LOGISTIC: logistic,
        KernelType.LOG_LOGISTIC: log_logistic,
        KernelType.COSINE: cosine,
        KernelType.SINC: sinc,
        KernelType.LAPLACE: laplace,
        KernelType.QUARTIC: quartic,
        KernelType.PARABOLIC: parabolic,
        KernelType.EXPONENTIAL: exponential,
        KernelType.SILVERMAN: silverman,
        KernelType.CAUCHY: cauchy,
        KernelType.TENT: tent,
        KernelType.WAVE: wave,
        KernelType.POWER: power,
        KernelType.MORTERS: morters,
    }
    return kernel_map.get(kernel_type, laplace)


class KernelRegression:
    """Класс для расчёта kernel regression"""
    
    def __init__(self, kernel_type: KernelType = KernelType.LAPLACE, bandwidth: int = 14):
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.kernel_func = get_kernel_function(kernel_type)
        self._weights = None
        self._sum_weights = None
        self._precalculate_weights()
    
    def _precalculate_weights(self):
        """Предрасчёт весов для non-repaint версии"""
        weights = []
        sum_weights = 0.0
        
        for i in range(self.bandwidth):
            # j = i^2 / bandwidth^2 (как в Pine Script)
            j = (i ** 2) / (self.bandwidth ** 2)
            weight = self.kernel_func(j, 1)
            weights.append(weight)
            sum_weights += weight
        
        self._weights = np.array(weights)
        self._sum_weights = sum_weights
    
    def calculate(self, prices: np.ndarray) -> float:
        """
        Рассчитать значение kernel regression для последней точки
        Non-repaint версия
        """
        if len(prices) < self.bandwidth:
            return prices[-1] if len(prices) > 0 else 0.0
        
        # Берём последние bandwidth значений
        # В Pine Script: sum += nz(src[i]) * weight, где i от 0 до bw-1
        # src[0] - текущая цена, src[1] - предыдущая
        recent_prices = prices[-self.bandwidth:]
        
        # Взвешенная сумма (recent_prices[::-1] дает [текущая, пред, ...])
        weighted_sum = np.sum(recent_prices[::-1] * self._weights)
        
        return weighted_sum / self._sum_weights
    
    def calculate_series(self, prices: np.ndarray) -> np.ndarray:
        """Рассчитать серию значений kernel regression"""
        result = np.full(len(prices), np.nan)
        
        for i in range(self.bandwidth - 1, len(prices)):
            window = prices[i - self.bandwidth + 1:i + 1]
            weighted_sum = np.sum(window[::-1] * self._weights)
            result[i] = weighted_sum / self._sum_weights
        
        return result