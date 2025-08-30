from datetime import datetime


__all__ = [
    "get_time_stamp",
]


def get_time_stamp(
    show_second: bool = False,
    show_minute: bool = False,
)-> str:
    
    if show_second:
        return datetime.now().strftime("%y%m%d_%H%M%S")
    
    elif show_minute:
        return datetime.now().strftime("%y%m%d_%H%M")
    
    else:
        return datetime.now().strftime("%y%m%d")