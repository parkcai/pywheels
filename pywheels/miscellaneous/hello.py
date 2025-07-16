from ..i18n import translate


__all__ = [
    "print_helloworld",
    "greet",
]


def print_helloworld(
)-> None:
    
    print(translate("Hello, World!"))
    
    
def greet(
    name: str,
)-> None:
    
    print(translate("Hello, %s!") % (name))