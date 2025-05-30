from setuptools import setup, find_packages

setup(
    name="foci",
    version="0.1.0",
    description="A Python package for foci project",
    author="Mario Gomez Andreu and Maximum Wilder-Smith",
    author_email="margomez@ethz.ch",
    packages=find_packages(),
    install_requires=["numpy",
        "open3d",
        "matplotlib",
        "viser",
        "plyfile",
        "scipy",
        "astar",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)