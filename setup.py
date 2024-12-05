# setup.py
from setuptools import setup, find_packages

setup(
    name="ad_agents",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        'fastapi==0.104.1',
        'langchain==0.0.300',
        'pydantic==1.10.13',
        'python-dotenv==1.0.0',
        'openai==0.27.0',
        'numpy==1.24.0',
        'pandas==2.0.0',
        'scikit-learn==1.3.2',
        'aiohttp==3.9.1'
    ]
)