__all__ = [
    "get_integer_input",
    "get_boolean_input",
    "get_string_input",
    "get_literal_input",
]


def get_integer_input(
    prompt: str, 
    default: int
) -> int:
    while True:
        try:
            user_input = input(f"{prompt} [default = {default}]: ").strip()
            if not user_input and default is not None:
                return default
            return int(user_input)
        except ValueError:
            print("请输入一个有效的整数！")


def get_boolean_input(
    prompt: str,
    default: bool
) -> bool:
    yes_set = {"yes", "y"}
    no_set = {"no", "n"}

    default_str = "Yes" if default else "No"

    while True:
        user_input = input(f"{prompt} [default = {default_str}]: ").strip().lower()

        if not user_input and default is not None:
            return default

        if user_input in yes_set:
            return True
        elif user_input in no_set:
            return False
        else:
            print("请输入 Yes 或 No（或 y / n）！")


def get_string_input(
    prompt: str,
    default: str
) -> str:
    while True:
        user_input = input(f"{prompt} [default = {default}]: ").strip()
        if not user_input and default is not None:
            return default
        if user_input:  # 确保不是空字符串
            return user_input
        print("请输入一个有效的字符串！")


def get_literal_input(
    prompt: str,
    default: str,
    options: list[str]
) -> str:
    options_lower = [opt.lower() for opt in options]
    default_str = f" [default = {default}]" if default is not None else ""
    options_prompt = f" ({'/'.join(options)})"

    while True:
        user_input = input(f"{prompt}{options_prompt}{default_str}: ").strip().lower()

        if not user_input and default is not None:
            return default

        if user_input in options_lower:
            # 返回原始大小写的选项
            return options[options_lower.index(user_input)]
        
        print(f"请输入有效的选项 ({'/'.join(options)})！")