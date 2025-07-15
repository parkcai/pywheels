# pywheels

Light-weight Python wheels for parkcai.

## 安装

您可以通过以下命令安装 `pywheels`：

```bash
pip install pywheels
```

## 贡献

欢迎贡献您的代码或建议！请在 GitHub 提交问题或拉取请求。

## 国际化

### 国际化初始化

- 使用标准库 `gettext` 实现多语言支持。
- 每个需要国际化的模块应在顶部显式导入：

```python
from gettext import gettext as translate
```

- 统一调用 `setup_translate_language()` 初始化语言环境，该函数在 `pywheels/i18n.py` 中定义并在`pywheels/__init__.py`中调用一次：

```python
# pywheels/__init__.py
from .i18n import setup_translate_language
setup_translate_language()
```

- `setup_translate_language()` 具备良好的跨平台兼容性，自动从系统环境变量中检测语言并加载翻译（见 `pywheels/i18n.py` 模块）。

### 生成国际化目标文件（.mo）的基本步骤

1. **提取所有 `.py` 文件中的翻译字符串，为目标语言准备 `.po` 文件（如尚未存在）并明确编码方式为 UTF-8**：

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

特别注意，参数`--keyword=translate`告诉了 xgettext 识别 `translate`。

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

- 所有待翻译内容需用 `translate("...")` 包裹，确保能被提取。
- `.po` 和 `.mo` 文件需为 UTF-8 编码。
- 项目只初始化一次语言环境，其它模块无需重复加载。
- 推荐使用 C 风格占位符（如 `%s`, `%d`）以便 `.po` 文件翻译更自然：

```python
translate("Hello, %s!") % name
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

---

### 注意事项

- 使用 API token 代替密码，安全且方便自动化。
- PyPI 不允许覆盖已发布的同版本文件，上传前务必更新版本号。
- PyPI 删除 release 时仅隐藏，不会物理删除版本号；已用版本号永久保留，无法重复上传。
