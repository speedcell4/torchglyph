from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fp:
    long_description = fp.read()

setup(
    name='torchglyph',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/speedcell4/torchglyph',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='Data processing utilities for natural language processing',
    long_description=long_description,
    install_requires=[
        'tqdm',
        'numpy',
    ],
    extras_require={
        'dev': [
            'pytest',
            'hypothesis',
        ],
    }
)
