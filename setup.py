from setuptools import setup, find_packages

name = 'torchglyph'

setup(
    name=name,
    version='0.1.1',
    packages=[package for package in find_packages() if package.startswith(name)],
    url=f'https://speedcell4.github.io/torchglyph',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='Data Processor Combinators for Natural Language Processing',
    install_requires=[
        'tqdm',
        'numpy',
        'einops',
    ],
    extras_require={
        'dev': [
            'pytest',
            'hypothesis',
        ],
        'ctx': [
            'transformers',
            'allennlp',
            'elmoformanylangs',
        ],
        'docs': [
            'mkdocs',
            'mkdocs-alabaster',
        ]
    }
)
