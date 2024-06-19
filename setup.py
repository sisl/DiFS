from setuptools import setup, find_packages

setup(
    name='difs',
    version='0.1.0',
    description = 'Diffusion-based Failure Sampling',
    author='Harrison Delecki',
    author_email='hdelecki@stanford.edu',
    url='https://github.com/sisl/DiFS',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'einops',
        'accelerate',
        'ema_pytorch>=0.4.2',
        'wandb',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)