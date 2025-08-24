import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from ..typing import *


__all__ = [
    "fourier_transform",
    "get_fourier_transform_amplitude",
    "moving_average",
    "normalize",
    "reshape_2d_numpy_array",
    "searchsorted_nearest",
    "check_ndarray",
    "compress_ndarray",
]


def fourier_transform(
    signal: ArrayLike,
    interval: float,
)-> Tuple[ndarray, ndarray]:
    
    """
    计算信号的傅里叶变换（包括幅度和对应频率）。

    参数:
        signal (ArrayLike): 一维或二维信号数组。二维数组按列为独立信号。
        interval (float): 采样时间间隔。

    返回:
        Tuple[ndarray, ndarray]: 傅里叶变换幅度和频率数组。
    """
    
    signal = np.asarray(signal)

    if signal.ndim == 1:
        transformed = np.fft.fft(signal)
        n = signal.shape[0]
    elif signal.ndim == 2:
        transformed = np.fft.fft(signal, axis=0)
        n = signal.shape[0]
    else:
        raise ValueError("输入信号必须是一维或二维数组。")

    magnitude = np.abs(transformed)
    freq = np.fft.fftfreq(n, d = interval)

    return magnitude, freq


def get_fourier_transform_amplitude(
    signal: ArrayLike,
)-> ndarray:
    
    """
    计算信号的傅里叶变换幅度。

    参数:
        signal (ArrayLike): 一维或二维信号数组。

    返回:
        ndarray: 幅度谱数组。
    """
    
    signal = np.asarray(signal)

    if signal.ndim == 1:
        transformed = np.fft.fft(signal)
    elif signal.ndim == 2:
        transformed = np.fft.fft(signal, axis=0)
    else:
        raise ValueError("输入信号必须是一维或二维数组。")

    return np.abs(transformed)


def moving_average(
    arr: ArrayLike,
    window_size: int,
)-> ndarray:
    
    """
    计算滑动平均。

    参数:
        arr (ArrayLike): 一维输入数据。
        window_size (int): 滑动窗口大小，通常为奇数。

    返回:
        ndarray: 与原始长度一致的滑动平均结果。
    """
    
    arr = np.asarray(arr)
    windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_size)
    moving_avg = np.mean(windows, axis=1)

    result = np.empty_like(arr, dtype=float)
    pad = window_size // 2
    result[:pad] = arr[:pad]
    result[pad:pad + len(moving_avg)] = moving_avg
    result[pad + len(moving_avg):] = arr[-pad:]
    return result


def normalize(
    arr: ArrayLike,
)-> ndarray:
    
    """
    将数组归一化到 [0, 1] 范围。

    参数:
        arr (ArrayLike): 输入数组。

    返回:
        ndarray: 归一化后的数组。
    """
    
    arr = np.asarray(arr)
    max_vector = np.max(arr, axis=0)
    min_vector = np.min(arr, axis=0)
    return (arr - min_vector) / (max_vector - min_vector)


def reshape_2d_numpy_array(
    original_matrix: ArrayLike,
    reshaped_row_num: int,
    reshaped_col_num: int
)-> ndarray:
    
    """
    重塑二维数组形状。

    参数:
        original_matrix (ArrayLike): 原始二维数组。
        reshaped_row_num (int): 目标行数。
        reshaped_col_num (int): 目标列数。

    返回:
        ndarray: 重塑后的二维数组。

    异常:
        ValueError: 若原始元素总数与目标不匹配。
    """
    
    original_matrix = np.asarray(original_matrix)
    if original_matrix.ndim != 2:
        raise ValueError("输入必须是二维数组。")

    if original_matrix.size != reshaped_row_num * reshaped_col_num:
        raise ValueError(f"原数组的总元素个数与目标形状不匹配，原数组形状为 {original_matrix.shape}，"
                         f"目标形状为 ({reshaped_row_num}, {reshaped_col_num})。")

    return original_matrix.reshape(reshaped_row_num, reshaped_col_num)


def searchsorted_nearest(
    arr: ArrayLike,
    value: float,
)-> int:
    
    """
    在升序数组中查找最接近给定值的索引。

    参数:
        arr (ArrayLike): 一维升序数组。
        value (float): 待查找的目标值。

    返回:
        int: 最接近值的索引。
    """
    arr = np.asarray(arr)
    idx = int(np.searchsorted(arr, value))  # <-- 关键在这显式转为 int

    if idx == 0:
        return 0
    if idx == len(arr):
        return len(arr) - 1

    return idx - 1 if abs(arr[idx - 1] - value) < abs(arr[idx] - value) else idx


def check_ndarray(
    arr: ArrayLike
)-> bool:
    
    """
    检查数组是否含有 NaN 或 Inf。

    参数:
        arr (ArrayLike): 输入数组。

    返回:
        bool: True 表示无 NaN/Inf，False 表示异常。
    """
    arr = np.asarray(arr)
    return not (np.isinf(arr).any() or np.isnan(arr).any())


def compress_ndarray(
    input_ndarray: ArrayLike,
    output_shape: Tuple[int, int]
)-> ndarray:
    
    """
    压缩二维数组，按行平均压缩。

    参数:
        input_ndarray (ArrayLike): 原始二维数组。
        output_shape (Tuple[int, int]): 目标形状 (rows, cols)，其中 rows 必须能整除原始行数。

    返回:
        ndarray: 压缩后的数组。

    异常:
        ValueError: 若维度不符或形状不合法。
    """
    input_ndarray = np.asarray(input_ndarray)

    if input_ndarray.ndim != 2:
        raise ValueError("input_ndarray 必须是二维的。")

    if output_shape[1] != input_ndarray.shape[1]:
        raise ValueError(f"列数不匹配：输入为 {input_ndarray.shape[1]}，目标为 {output_shape[1]}。")

    k = input_ndarray.shape[0] // output_shape[0]
    if k * output_shape[0] != input_ndarray.shape[0]:
        raise ValueError("行数不匹配：输入行数必须是目标行数的整数倍。")

    return input_ndarray.reshape(output_shape[0], k, output_shape[1]).mean(axis=1)
