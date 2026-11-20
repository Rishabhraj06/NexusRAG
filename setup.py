from setuptools import setup, find_packages

setup(
    name="nexusrag",
    version="1.0.0",
    author="Rishabh Raj",
    author_email="rishabhraj74918@gmail.com",
    description="Enterprise Multi-Agent Hybrid RAG Engine with Self-Reflective Retrieval",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Rishabhraj06/NexusRAG",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "pydantic>=2.0.0",
        "streamlit>=1.35.0",
    ],
    extras_require={
        "pdf": ["pypdf>=4.0.0"],
        "ai": ["google-genai>=0.1.0"],
        "dev": ["pytest>=8.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
