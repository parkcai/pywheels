__all__ = [
    "is_same_type",
    "is_hashable",
]


def is_same_type(
    obj1, 
    obj2,
)-> bool:
    
    return type(obj1) is type(obj2)


def is_hashable(
    obj,
)-> bool:
    
    try:
        
        hash(obj)
        return True
    
    except TypeError:
        return False
