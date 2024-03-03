from pathlib import Path

from setuptools import find_packages, setup

name = 'torchglyph'

root_dir = Path(__file__).parent.resolve()
with (root_dir / 'requirements.txt').open(mode='r', encoding='utf-8') as fp:
    install_requires = [install_require.strip() for install_require in fp]

setup(
    name=name,
    version='0.4.0',
    packages=[package for package in find_packages() if package.startswith(name)],
    url=f'https://speedcell4.github.io/torchglyph',
    license='MIT',
    author='speedcell4',
    author_email='speedcell4@gmail.com',
    description='Data Processor Combinators for Natural Language Processing',
    python_requires='>=3.9',
    install_requires=install_requires,
)
