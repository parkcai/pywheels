from setuptools import setup
from setuptools import find_packages


setup(
    name = 'pywheels',
    version = '0.3.0',
    packages = find_packages(),
    description = 'Light-weight Python wheels',
    author = 'parkcai',
    author_email = 'sun_retailer@163.com',
    url = 'https://github.com/parkcai/pywheels',
    include_package_data = True,
    package_data = {
        'pywheels': ['locales/**/LC_MESSAGES/*.mo'],
    },
)