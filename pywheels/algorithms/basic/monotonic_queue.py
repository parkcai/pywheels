import operator
from typing import List
from typing import TypeVar
from typing import Sequence
from typing import Callable
from typing import Protocol
from typing import runtime_checkable
from collections import deque
from ...i18n import translate


__all__ = [
    "sliding_window_greatest",
    "sliding_window_smallest",
]


@runtime_checkable
class SupportsRichComparison(Protocol):
    def __gt__(self, other)-> bool: ...
    def __eq__(self, other)-> bool: ...
    def __lt__(self, other)-> bool: ...


DatumType = TypeVar("DatumType")
KeyType = TypeVar("KeyType", bound = SupportsRichComparison)


def _sliding_window_best(
    data: Sequence[DatumType],
    window_size: int,
    key: Callable[[DatumType], KeyType],
    best_means_greatest: bool,
)-> List[DatumType]:
    
    _sliding_window_best_entrance_check(
        data = data,
        window_size = window_size,
        key = key,
    )
    
    def is_better_than(
        datum1: DatumType,
        datum2: DatumType,
    )-> bool:
        
        datum1_key = key(datum1)
        datum2_key = key(datum2)
        
        return (best_means_greatest and datum2_key > datum1_key) \
            or (not best_means_greatest and datum2_key < datum1_key)
            
    possible_best_students = deque()
    results = []
    
    for student, datum in enumerate(data):
        
        # 老生毕业
        while possible_best_students \
            and student - possible_best_students[0] >= window_size:
                
            possible_best_students.popleft()
        
        # 新生入学，赶走又老又菜的
        while possible_best_students \
            and is_better_than(data[possible_best_students[-1]], datum):
                
            possible_best_students.pop()
            
        possible_best_students.append(student)
        
        # 出结果
        results.append(data[possible_best_students[0]])
        
    return results


def _sliding_window_best_entrance_check(
    data: Sequence[DatumType],
    window_size: int,
    key: Callable[[DatumType], KeyType],
)-> None:
    
    if not data:
        
        raise ValueError(
            translate("参数 `data` 不能为空！")
        )
        
    if window_size <= 0:
        
        raise ValueError(
            translate("参数 `window_size` 应为正整数！")
        )
        
    if window_size > len(data):
        
        raise ValueError(
            translate("参数 `window_size` 应小于参数 `data` 的长度！")
        )
        
    first_datum_key = key(data[0])
    
    try:
        
        operator.gt(first_datum_key, first_datum_key)
        operator.eq(first_datum_key, first_datum_key)
        operator.lt(first_datum_key, first_datum_key)
        
    except TypeError:
        
        raise RuntimeError(
            translate("参数 `key` 的返回值应支持大于、等于和小于运算符！")
        )


def sliding_window_greatest(
    data: Sequence[DatumType],
    window_size: int,
    key: Callable[[DatumType], KeyType] = lambda x: x, # type: ignore
)-> List[DatumType]:
    
    return _sliding_window_best(
        data = data,
        window_size = window_size,
        key = key,
        best_means_greatest = True,
    )
    
    
def sliding_window_smallest(
    data: Sequence[DatumType],
    window_size: int,
    key: Callable[[DatumType], KeyType] = lambda x: x, # type: ignore
)-> List[DatumType]:
    
    return _sliding_window_best(
        data = data,
        window_size = window_size,
        key = key,
        best_means_greatest = False,
    )


