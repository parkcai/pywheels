__all__ = [
    "is_same_type",
]


def is_same_type(
    a, 
    b,
)-> bool:
    
    return type(a) is type(b)