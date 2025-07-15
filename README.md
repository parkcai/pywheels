# pywheels

Light-weight Python wheels for parkcai.

## 安装

您可以通过以下命令安装 `pywheels`：

```bash
pip install pywheels
```

## 贡献

欢迎贡献您的代码或建议！请在 GitHub 提交问题或拉取请求。

## 备忘：发布到 PyPI 的方式

### 构建包

在项目根目录下运行以下命令以生成分发文件：

```bash
python setup.py sdist bdist_wheel
```

### 登录 PyPI

尚未注册账号可访问 [PyPI](https://pypi.org/account/register/) 创建账户。

安装 Twine（如尚未安装）：

```bash
pip install twine
```

建议使用 API token 登录，更安全：

```bash
twine upload dist/*
```

首次执行 `upload` 时将提示输入用户名和密码。推荐在 `~/.pypirc` 中配置：

```ini
[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 上传分发包

执行上传命令：

```bash
twine upload dist/*
```
