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


class DynamicTranslate:
    
    def __init__(self):
        self._func: Callable[[str], str] = gettext.gettext

    def __call__(self, msg: str) -> str:
        return self._func(msg)

    def bind(self, func: Callable[[str], str]):
        self._func = func


translate = DynamicTranslate()


def set_language(
    language: str,
)-> None:
    
    global translate
    
    domain = "messages"
    localedir = str(Path(__file__).resolve().parent / "locales")
    
    if language == "en": language = "en_US"
    if language == "zh": language = "zh_CN"
    
    if language == "zh_TW": 
        print("zh_TW is not supported. Use zh instead.")
        language = "zh"

    t = gettext.translation(
        domain = domain,
        localedir = localedir,
        languages = [language],
        fallback = True,
    )
    
    translate.bind(t.gettext)


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
        language = "en"
        
    language = language.split('.')[0]
    
    if language == "zh_TW": 
        print("zh_TW is not supported. Use zh instead.")
        language = "zh"
    
    set_language(language)

     
try:
    init_language()
except Exception as _:
    pass
