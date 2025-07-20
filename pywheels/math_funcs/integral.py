from typing import Callable
from scipy.integrate import quad


__all__ = [
    "integral_1d_func",
]


def integral_1d_func(
    func: Callable[[float], float],
    start: float,
    end: float,
    absolute_error: float = 1.49e-8,
    relative_error: float = 1.49e-8,
)-> float:
    
    """
    计算一维函数在指定区间内的定积分。

    使用 SciPy 的 quad 方法进行数值积分，允许用户设置绝对误差和相对误差控制精度。

    参数：
        func (Callable[[float], float]): 被积函数，接受一个 float 类型参数并返回 float。
        start (float): 积分区间起点。
        end (float): 积分区间终点。
        absolute_error (float): 积分允许的绝对误差容限。
        relative_error (float): 积分允许的相对误差容限。

    返回值：
        float: 计算得到的积分值。
    """
    
    integral_result = quad(
        func = func, 
        a = start,
        b = end,
        epsabs = absolute_error, 
        epsrel = relative_error,
    )[0]
    
    return integral_result