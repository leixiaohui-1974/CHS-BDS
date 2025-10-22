"""
Setup script for CHS-BDS
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='chs-bds',
    version='0.1.0',
    author='Lei Xiaohui',
    author_email='leixiaohui@example.com',
    description='GNSS Comprehensive Monitoring System for environmental and geological hazard monitoring',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/leixiaohui-1974/CHS-BDS',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=23.0',
            'flake8>=6.0',
            'mypy>=1.0',
        ],
        'viz': [
            'plotly>=5.0',
            'dash>=2.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'chs-bds=gnss_monitoring.cli:main',
        ],
    },
    include_package_data=True,
    keywords='GNSS GPS BeiDou monitoring deformation rainfall PWV meteorology',
    project_urls={
        'Bug Reports': 'https://github.com/leixiaohui-1974/CHS-BDS/issues',
        'Source': 'https://github.com/leixiaohui-1974/CHS-BDS',
        'Documentation': 'https://github.com/leixiaohui-1974/CHS-BDS#readme',
    },
)
