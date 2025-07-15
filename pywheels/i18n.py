import os
import locale
import gettext
from pathlib import Path
from typing import Optional


def setup_translate_language(
    domain: str = "messages",               # 默认为 'messages'，这是 .mo 文件的默认基名
    localedir: Optional[str] = None,        # 指定翻译文件目录，默认 None，稍后自动设置
):
    # 优先从环境变量中获取语言设置，适配 Linux/macOS 的常用变量
    lang = (
        os.getenv("LANG") or                # 常见格式如 "zh_CN.UTF-8"
        os.getenv("LC_ALL") or              # 全局语言优先级高于 LANG
        os.getenv("LANGUAGE")               # 某些系统使用 LANGUAGE 覆盖 LANG
    )

    if not lang:
        # 如果环境变量中没有定义语言，就使用 Python locale 模块检测系统默认语言
        lang, _ = locale.getdefaultlocale()  # 返回如 ("zh_CN", "UTF-8")

    # 去掉后缀（如 UTF-8），只保留语言代码部分（如 zh_CN）
    lang = (lang or "en_US").split('.')[0]

    if localedir is None:
        # 如果没有手动指定 locales 路径，就使用当前文件旁边的 "locales" 文件夹
        localedir = str(Path(__file__).resolve().parent / "locales")

    try:
        # 尝试加载指定语言的翻译文件 (.mo)
        gettext.translation(
            domain=domain,                   # 如 messages.mo
            localedir=localedir,             # 语言目录路径
            languages=[lang],                # 语言列表，只取一个
            fallback=True                    # 若找不到翻译文件则返回原始字符串
        )
        # 注意：此处没有调用 install()，因为每个模块用 from gettext import gettext as translate
        # 所以只需调用 translation() 确保 gettext 系统初始化，无需全局绑定 _
    except Exception as error:
        # 如果语言文件加载失败，打印一个警告（不抛出异常，保证程序健壮性）
        print(
            f"[warn] Failed to load translation for language '{lang}':\n"
            f"{error}"
        )
