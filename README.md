# pywheels

Light-weight Python wheels.

## 安装

您可以通过以下命令安装 `pywheels`：

```bash
pip install pywheels
```

## 功能

### llm_tools

- get_answer：方便调大语言模型 API 的一个 wrapper 函数

## 贡献

欢迎贡献您的代码或建议！请在 GitHub 提交问题或拉取请求。

## 国际化

### 基本情况

- 使用标准库 `gettext` 实现多语言支持。
- 项目在 `pywheels/i18n.py` 中定义了三个接口：

  - `translate`: 翻译函数，模块内可用 `from pywheels.i18n import translate` 导入（推荐使用相对路径）；
  - `set_language(language)`: 手动切换语言；
  - `init_language()`: 从系统环境变量自动初始化语言（已在模块加载时默认执行）。

- 示例用法：

```python
# pywheels/miscellaneous/hello.py
from ..i18n import translate

def print_helloworld(
)-> None:
    print(translate("Hello, World!"))
```

- `translate()` 默认会根据环境变量（如 `LANG`, `LC_ALL`）自动选择语言。也可通过调用 `set_language('zh')` 来手动切换语言。

### 生成国际化目标文件（.mo）的基本步骤

1. **提取`pywheels`中所有 `.py` 文件中的翻译字符串，为目标语言准备 `.po` 文件（如尚未存在）并明确编码方式为 UTF-8**：

```bash
xgettext -L Python --keyword=translate -o pywheels/locales/messages.pot $(find . -name "*.py")

for lang in zh_CN en_US; do
  mkdir -p pywheels/locales/$lang/LC_MESSAGES
  [ -f pywheels/locales/$lang/LC_MESSAGES/messages.po ] || cp pywheels/locales/messages.pot pywheels/locales/$lang/LC_MESSAGES/messages.po
done

for lang in zh_CN en_US; do
  sed -i 's/charset=CHARSET/charset=UTF-8/' pywheels/locales/$lang/LC_MESSAGES/messages.po
done
```

> 💡 参数 `--keyword=translate` 告诉 `xgettext` 将 `translate("...")` 作为翻译目标。

2. **翻译 `.po` 文件中的内容**

使用文本编辑器（如 VS Code）、翻译软件（如 Poedit）或命令行工具编辑每个 `.po` 文件，填入对应语言的翻译：

```po
msgid "Hello, %s!"
msgstr "你好，%s！"
```

3. **批量编译 `.po` 为 `.mo` 文件**：

```bash
for lang in zh_CN en_US; do
  msgfmt pywheels/locales/$lang/LC_MESSAGES/messages.po -o pywheels/locales/$lang/LC_MESSAGES/messages.mo
done
```

### 注意事项

- 所有待翻译内容需用 `translate("...")` 包裹，确保能被 `xgettext` 提取。
- `.po` 和 `.mo` 文件需为 UTF-8 编码。
- 国际化模块仅初始化一次语言环境，其它模块通过导入 `translate` 使用即可。
- 推荐使用 C 风格占位符（如 `%s`, `%d`），以便 `.po` 文件中的翻译更自然、格式更稳定：

```python
translate("Hello, %s!") % (name)
```

## 发布至 PyPI

### 构建包

确保修改 `setup.py` 中的版本号后，在项目根目录下运行：

```bash
rm -rf build/ dist/ *.egg-info && python setup.py sdist bdist_wheel
```

### 登录 PyPI

尚未注册账号可访问 [PyPI](https://pypi.org/account/register/) 创建账户。

安装 Twine（如尚未安装）：

```bash
pip install twine
```

建议使用 API token 登录，更安全。推荐在 `~/.pypirc` 中配置：

```ini
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

首次上传时会提示输入用户名和密码，配置后可免交互。

### 上传分发包

执行上传命令：

```bash
twine upload dist/*
```

### 注意事项

- 使用 API token 代替密码，安全且方便自动化。
- PyPI 不允许覆盖已发布的同版本文件，上传前务必更新版本号。
- PyPI 删除 release 时仅隐藏，不会物理删除版本号；已用版本号永久保留，无法重复上传。
