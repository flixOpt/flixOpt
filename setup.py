from setuptools import setup, find_packages

def read_requirements(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]
#TODO: Change Autor Name and email
setup(
    name='flixOpt',
    version='0.1.0',
    python_requires='>=3.9, <3.12',
    author='Felix Bumann',
    author_email='felixbumann387@gmail.com',
    description='Vector based energy and material flow optimization framework in python.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/flixOpt/flixOpt',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Topic :: Scientific/Engineering :: Mathematics :: Numerical Analysis :: Optimization',
        'Topic :: Software Development :: Libraries :: Python'
    ],

    packages=find_packages(exclude=['tests', 'docs', 'examples', 'examples.*'
                                    '.git', '.vscode', 'build', '.venv', 'venv/',
                                    ]),
    install_requires=read_requirements('requirements.txt')
)