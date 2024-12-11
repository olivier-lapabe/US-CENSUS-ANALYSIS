from setuptools import setup, find_packages

setup(
    name="US-CENSUS-ANALYSIS",
    packages=find_packages(),
    install_requires=[
        'lightgbm==4.5.0',
        'xgboost==2.1.3',
        'pandas==2.2.3',
        'numpy==2.2.0',
        'matplotlib==3.9.3',
        'seaborn==0.13.2',
        'scipy==1.14.1',
        'prince==0.14.0',
        'scikit-learn==1.5.1',
    ]
)
