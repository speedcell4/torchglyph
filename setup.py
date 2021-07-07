from setuptools import setup, find_packages

name = 'torchglyph'

setup(
    name=name,
    version='0.1.2',
    packages=[package for package in find_packages() if package.startswith(name)],
    url=f'https://speedcell4.github.io/torchglyph',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='Data Processor Combinators for Natural Language Processing',
    install_requires=[
        'aku',
        'tqdm',
        'numpy',
        'einops',
        'torchrua',
        'requests',
    ],
    extras_require={
        'dev': [
            'pytest',
            'hypothesis',
        ],
        'ctx': [
            'transformers',
        ],
        'docs': [
            'mkdocs',
            'mkdocs-alabaster',
        ]
    }
)
