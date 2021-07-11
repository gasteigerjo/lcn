from setuptools import setup

install_requires_lcn = [
    "munch",
    "numpy",
    "scipy>=1.3",
    "torch==1.4",  # norm broken in 1.5, some torchscript stuff broken later (v1.8, 1.9)
    "torch_scatter",
    "sacred",
    "seml",
]

setup(
    name="LCN",
    version="1.0",
    description="Locally Corrected Nyström",
    author="Johannes Klicpera",
    author_email="klicpera@in.tum.de",
    packages=["lcn"],
    install_requires=install_requires_lcn,
    zip_safe=False,
)
