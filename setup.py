from setuptools import setup, find_packages

setup(
    name='herding',
    version='0.0.1',
    install_requires=['gym', 'numpy', 'pyopencl'],
    packages=[package for package in find_packages()
              if package.startswith('herding')],
    package_data={
        '': ['*.cu', '*.cuh', '*.json']
    }
)
