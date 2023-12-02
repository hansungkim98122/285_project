from setuptools import setup
import sys

setup(
    name='TeachMyAgent',
    py_modules=['TeachMyAgent'],
    version="1.0",
    install_requires=[
        'cloudpickle==1.2.0',
        'gym[atari,box2d,classic_control]>=0.10.8',
        'ipython',
        'joblib',
        'matplotlib',
        'numpy',
        'pytest',
        'psutil',
        'imageio',
        'seaborn==0.8.1',
        'dm-sonnet<2',
        'setuptools',
        'setuptools_scm',
        'pep517',
        'treelib',
        'gizeh',
        'tqdm',
        'emcee',
        'notebook',
        'huggingface_hub',
    ],
    description="TeachMyAgent: A benchmark to study and compare ACL algorithms for DeepRL in continuous procedural environments.",
    author="Clément Romac",
)

