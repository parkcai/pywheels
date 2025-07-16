import os
import locale
import gettext
from pathlib import Path
from typing import Callable


__all__ = [
    "translate",
    "set_language",
    "init_language",
]

        
translate: Callable[[str], str] = gettext.gettext


def set_language(
    language: str,
)-> None:
    
    global translate
    
    domain = "messages"
    localedir = str(Path(__file__).resolve().parent / "locales")

    t = gettext.translation(
        domain = domain,
        localedir = localedir,
        languages = [language],
        fallback = True,
    )
    
    translate =  t.gettext


def init_language(
)-> None:
    
    language = (
        os.getenv("LANG") or
        os.getenv("LC_ALL") or
        os.getenv("LANGUAGE")
    )

    if not language:
        language, _ = locale.getdefaultlocale()
        
    if not language:
        language = "en_US"
        
    language = language.split('.')[0]
    
    set_language(language)

     
try:
    init_language()
except Exception as _:
    pass
