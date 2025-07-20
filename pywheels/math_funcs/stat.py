import numpy as np
from numpy.typing import NDArray
from typing import Any
from ..i18n import translate


__all__ = [
    "chi_squared",
    "mean_squared_error",
]


def _check_dtype_valid(
    arr: NDArray[Any], 
    name: str
)-> None:
    
    """
    确保数组的数据类型是 int 或 float
    """
    
    if not np.issubdtype(arr.dtype, np.integer) \
        and not np.issubdtype(arr.dtype, np.floating):
            
        raise TypeError(
            translate(
                f"%s 的数据类型必须为整数或浮点数，当前为 %s"
                % (name, str(arr.dtype))
            )
        )


def chi_squared(
    predicted_data: NDArray[Any],
    ground_truth_data: NDArray[Any],
    errors: NDArray[Any],
)-> float:
    
    predicted_data = np.asarray(predicted_data)
    ground_truth_data = np.asarray(ground_truth_data)
    errors = np.asarray(errors)

    _check_dtype_valid(predicted_data, translate("预测值"))
    _check_dtype_valid(ground_truth_data, translate("真实值"))
    _check_dtype_valid(errors, translate("误差"))

    if not (len(predicted_data) == len(ground_truth_data) == len(errors)):
        raise ValueError(translate("预测值、真实值和误差数组的长度必须一致。"))

    if np.any(errors <= 0):
        raise ValueError(translate("所有误差值必须为正数且非零。"))

    chi2 = np.sum(((predicted_data - ground_truth_data) / errors) ** 2)
    return float(chi2)


def mean_squared_error(
    predicted_data: NDArray[Any],
    ground_truth_data: NDArray[Any],
)-> float:
    
    predicted_data = np.asarray(predicted_data)
    ground_truth_data = np.asarray(ground_truth_data)

    _check_dtype_valid(predicted_data, translate("预测值"))
    _check_dtype_valid(ground_truth_data, translate("真实值"))

    if len(predicted_data) != len(ground_truth_data):
        raise ValueError(translate("预测值和真实值数组的长度必须一致。"))

    mse = np.mean((predicted_data - ground_truth_data) ** 2)
    return float(mse)
