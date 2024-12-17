from setuptools import setup, find_packages
import pathlib

# 获取当前目录的路径
HERE = pathlib.Path(__file__).parent

# 读取 README 文件作为长描述
long_description = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="PEM4Fe",
    version="1.0.2",
    description="A tool for data processing and peak detection for PEM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jiahui Zhai",
    author_email="19720212203881@stu.xmu.edu.cn",
    url="https://github.com/HUSKYzjh/PEM4Fe",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "PEM4Fe = PEM4Fe.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # 根据您的实际许可协议调整
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
