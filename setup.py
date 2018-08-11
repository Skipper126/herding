from setuptools import setup, find_packages

setup(
    name='herding',
    version='0.0.1',
    install_requires=['gym', 'numpy', 'pycuda'],
    packages=find_packages()
)
