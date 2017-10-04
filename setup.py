from setuptools import setup, find_packages

setup(name='keras_generator',
    version='0.1',
    packages=find_packages(),
    description='running cdiscount trainer on gcloud',
    author='sparky_005',
    install_requires=[
        'keras',
        'numpy',
        'pandas',
        'pymongo',
        'tqdm'
    ],
    zip_safe=False)
