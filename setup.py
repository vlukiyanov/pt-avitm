from setuptools import setup


setup(
    name='ptavitm',
    version='1.0',
    description='PyTorch implementation of AVITM.',
    author='Vladimir Lukiyanov',
    author_email='vladimir.lukiyanov@me.com',
    url='https://github.com/vlukiyanov/pt-avitm/',
    download_url='',
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=[
        'numpy>=1.13.3',
        'torch>=1.0.0',
        'scipy>=1.0.0',
        'pandas>=0.21.0',
        'click>=6.7',
        'cytoolz>=0.9.0.1',
        'tqdm>=4.11.2',
        'scikit-learn>=0.19.1',
        'tensorboardX>=1.2',
        'setuptools>=40.2.0',
        'torchvision>=0.2.1'
    ],
    packages=['ptavitm']
)
