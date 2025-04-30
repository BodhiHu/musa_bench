from setuptools import setup, find_packages

setup(
    name="musa_bench",
    version="0.1.0",
    description="YOLO live detection and benchmarking tools",
    author="MT Dev",
    packages=find_packages(),
    install_requires=[
        # "torch",
        # "opencv-python",
        # "fastapi",
    ],
    python_requires=">=3.7",
)
