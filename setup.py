from setuptools import find_packages, setup




setup(
    name="DenoiseSUM",
    packages=find_packages(),
    version="0.2.1",
    description="Denoisesum",
    author="",
    install_requires=[
        "torch",
        "ijson",
        "tqdm",
    ],
    license="MIT",

)
