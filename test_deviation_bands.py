"""Тест для проверки Deviation Bands"""
import numpy as np
from trading_bot.kernels import KernelRegression, KernelType


def test_deviation_bands():
    """Тест расчёта Deviation Bands"""
    # Создаём тестовые данные
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100) * 0.1) + 100
    
    # Тест с разными значениями deviations
    for deviations in [1.0, 2.0, 3.0]:
        kernel = KernelRegression(
            kernel_type=KernelType.LAPLACE,
            bandwidth=14,
            deviations=deviations
        )
        
        kernel_ma, upper_band, lower_band = kernel.calculate_with_bands(prices)
        
        # Проверяем, что полосы симметричны относительно MA
        diff_upper = upper_band[~np.isnan(upper_band)] - kernel_ma[~np.isnan(kernel_ma)]
        diff_lower = kernel_ma[~np.isnan(kernel_ma)] - lower_band[~np.isnan(lower_band)]
        
        print(f"\n=== Deviations = {deviations} ===")
        print(f"Kernel MA (last 5): {kernel_ma[-5:]}")
        print(f"Upper Band (last 5): {upper_band[-5:]}")
        print(f"Lower Band (last 5): {lower_band[-5:]}")
        print(f"Symmetry check (upper - lower): {np.allclose(diff_upper, diff_lower)}")
        print(f"Band width: {diff_upper[-1]:.6f}")
        
        # Проверяем, что upper > ma > lower
        valid_idx = ~np.isnan(kernel_ma)
        assert np.all(upper_band[valid_idx] >= kernel_ma[valid_idx]), "Upper band should be >= MA"
        assert np.all(lower_band[valid_idx] <= kernel_ma[valid_idx]), "Lower band should be <= MA"
    
    print("\n✅ Все тесты Deviation Bands пройдены!")


def test_calculate_series():
    """Тест обратной совместимости calculate_series"""
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(50) * 0.1) + 100
    
    kernel = KernelRegression(
        kernel_type=KernelType.EPANECHNIKOV,
        bandwidth=10,
        deviations=2.0
    )
    
    # Старый метод
    series_old = kernel.calculate_series(prices)
    
    # Новый метод (берём только MA)
    series_new, _, _ = kernel.calculate_with_bands(prices)
    
    # Должны быть идентичны
    assert np.allclose(series_old, series_new, equal_nan=True), "calculate_series и calculate_with_bands[0] должны совпадать"
    
    print("✅ Обратная совместимость calculate_series подтверждена!")


if __name__ == "__main__":
    test_deviation_bands()
    test_calculate_series()
