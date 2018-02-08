# encoding: utf-8
from setuptools import setup

# prepare long_description for PyPI:
long_description = None
try:
    long_description = open('README.rst').read()
    long_description += '\n' + open('CHANGES.rst').read()
except IOError:
    pass

setup(
    name='interpol',
    version='0.1.0',
    description='Simple CLI tools for analysis of pepperpot files',
    long_description=long_description,
    author='Thomas Gläßle',
    author_email='t_glaessle@gmx.de',
    maintainer='Thomas Gläßle',
    maintainer_email='t_glaessle@gmx.de',
    url='https://github.com/hibtc/interpol',
    license='GPLv3+',
    py_modules=['interpol', 'pepperpot', 'util'],
    entry_points={
        'console_scripts': [
            'pepperpot = pepperpot:main'
        ]
    },
    install_requires=[
        'docopt',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Physics'
    ],
)
