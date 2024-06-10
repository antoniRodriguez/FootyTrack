from setuptools import setup, find_packages

setup(
    name='footytracker',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'ultralytics',
        'opencv-python',
        'numpy',
        'pyyaml',
        'pytest',
        'pytest-mock',
        'supervision',
        'tensorflow',
        'PyYAML'
    ],
    entry_points={
        'console_scripts': [
            'footytracker = footballtracker.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['configs/*.yaml'],
    },
)
