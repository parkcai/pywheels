from gettext import gettext as translate


__all__ = [
    "print_helloworld",
]


def print_helloworld(
)-> None:
    
    print(translate("Hello, World!"))