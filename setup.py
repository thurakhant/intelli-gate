from setuptools import setup, find_packages

setup(
    name="model-lens",
    version="0.1.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'openai',
        'anthropic',
        'python-dotenv',
        'tiktoken'
    ],
    author="coTe",
    description="AI Model Gateway and Usage Tracking Platform",
    long_description=open('README.md').read() if open('README.md').read() else '',
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
)