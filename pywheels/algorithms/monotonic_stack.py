import operator
from typing import List
from typing import Union
from typing import TypeVar
from typing import Sequence
from typing import Callable
from typing import Protocol
from typing import runtime_checkable
from ..type_tools.basic import is_same_type
from ..i18n import translate


__all__ = [
    "next_greater_element",
    "next_smaller_element",
]


@runtime_checkable
class SupportsRichComparison(Protocol):
    def __gt__(self, other)-> bool: ...
    def __eq__(self, other)-> bool: ...
    def __lt__(self, other)-> bool: ...


DatumType = TypeVar("DatumType")
KeyType = TypeVar("KeyType", bound = SupportsRichComparison)


def _next_qualified_element(
    data: Sequence[DatumType],
    key: Callable[[DatumType], KeyType],
    right_direction: bool,
    strict: bool,
    return_index: bool,
    fill_factory: Callable[[], Union[int, DatumType]],
    qualified_means_greater: bool,
)-> Union[List[int], List[DatumType]]:
    
    _next_qualified_element_entrance_check(
        data = data,
        key = key,
        return_index = return_index,
        fill_factory = fill_factory,
    )
    
    key_dict = {}
    
    def is_qualified_for(
        datum1: DatumType, 
        datum2: DatumType,
    )-> bool:
        
        if datum1 not in key_dict: key_dict[datum1] = key(datum1)
        if datum2 not in key_dict: key_dict[datum2] = key(datum2)
        
        datum1_key = key_dict[datum1]
        datum2_key = key_dict[datum2]
        
        return (qualified_means_greater and datum2_key > datum1_key) \
            or (not qualified_means_greater and datum2_key < datum1_key) \
            or (not strict and datum2_key == datum1_key)
     
    index_to_answer = {}
    monotonic_stack = []
    
    data_length = len(data)
    
    for i in range(data_length):
        
        if right_direction:
            index = i
            
        else:
            index = data_length - 1 - i
            
        while monotonic_stack \
            and is_qualified_for(data[monotonic_stack[-1]], data[index]):
                
            index_to_answer[monotonic_stack.pop()] = index
            
        monotonic_stack.append(index)
        
    results = []
    
    for index in range(data_length):
        
        if index in index_to_answer:
            
            if return_index:
                result = index_to_answer[index]
                
            else:
                result = data[index_to_answer[index]]
            
        else:
            result = fill_factory()
            
        results.append(result)
            
    return results

    
def _next_qualified_element_entrance_check(
    data: Sequence[DatumType],
    key: Callable[[DatumType], KeyType],
    return_index: bool,
    fill_factory: Callable[[], Union[int, DatumType]],
)-> None:
    
    if not data:
        
        raise ValueError(
            translate("参数 `data` 不能为空！")
        )
        
    first_datum = data[0]
    test_fill = fill_factory()
    
    if return_index and not isinstance(test_fill, int):
        
        raise ValueError(
            translate("返回索引时，参数 `fill_factory` 的返回值应为整数！")
        )
        
    if not return_index and not is_same_type(test_fill, first_datum):
            
        raise ValueError(
            translate("返回数据时，参数 `fill_factory` 的返回值应与数据类型相同！")
        )
        
    first_datum_key = key(first_datum)
       
    try:
        
        operator.gt(first_datum_key, first_datum_key)
        operator.eq(first_datum_key, first_datum_key)
        operator.lt(first_datum_key, first_datum_key)
        
    except:
        
        raise RuntimeError(
            translate("参数 `key` 的返回值应支持大于、等于和小于运算符！")
        )


def next_greater_element(
    data: Sequence[DatumType],
    key: Callable[[DatumType], KeyType] = lambda x: x, # type: ignore
    right_direction: bool = True,
    strict: bool = True,
    return_index: bool = False,
    fill_factory: Callable[[], Union[int, DatumType]] = lambda : -1,
)-> Union[List[int], List[DatumType]]:
    
    return _next_qualified_element(
        data = data,
        key = key,
        right_direction = right_direction,
        strict = strict,
        return_index = return_index,
        fill_factory = fill_factory,
        qualified_means_greater = True,
    )
    
    
def next_smaller_element(
    data: Sequence[DatumType],
    key: Callable[[DatumType], KeyType] = lambda x: x, # type: ignore
    right_direction: bool = True,
    strict: bool = True,
    return_index: bool = False,
    fill_factory: Callable[[], Union[int, DatumType]] = lambda : -1,
)-> Union[List[int], List[DatumType]]:
    
    return _next_qualified_element(
        data = data,
        key = key,
        right_direction = right_direction,
        strict = strict,
        return_index = return_index,
        fill_factory = fill_factory,
        qualified_means_greater = False,
    )