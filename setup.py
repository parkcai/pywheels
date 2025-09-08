from setuptools import setup
from setuptools import find_packages


setup(
    name = "pywheels",
    version = "0.7.4.1",
    packages = find_packages(),
    description = "Light-weight Python wheels",
    author = "parkcai",
    author_email = "sun_retailer@163.com",
    url = "https://github.com/parkcai/pywheels",
    include_package_data = True,
    package_data = {
        "pywheels": ["locales/**/LC_MESSAGES/*.mo"],
    },
    python_requires = ">=3.8",
    install_requires = [
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "scipy>=1.7.3",
        "openpyxl>=3.0.10",
        "pandas>=1.5.0",
        "astor>=0.8.0",
    ],
)