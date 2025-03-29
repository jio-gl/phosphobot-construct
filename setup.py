from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phosphobot-construct",
    version="0.1.0",
    author="Phosphobot Team",
    author_email="info@phosphobot.ai",
    description="Robot Training with GenAI for Superhuman Skills",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phosphobot/phosphobot-construct",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "opencv-python>=4.5.0",
        "phosphobot>=0.1.0",
        "openai>=1.0.0",
        "transformers>=4.20.0",
        "pytorch3d>=0.7.0",
        "stable-baselines3>=2.0.0",
        "trimesh>=3.9.0",
        "pyBullet>=3.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "pillow>=9.0.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.9.0",
        ],
        "full": [
            "segment-anything>=1.0",
            "clip>=0.2.0",
            "diffusers>=0.18.0",
            "gymnasium>=0.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phosphobot-construct=phosphobot_construct.cli:main",
        ],
    },
)