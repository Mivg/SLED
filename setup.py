from setuptools import setup, find_packages
# This setup file was created with the help of https://github.com/MichaelKim0407/tutorial-pip-package

extra_t5 = [
    'protobuf<=3.20.1',
    'sentencepiece'
]
extra_examples = [
    'nltk',
    'datasets>=1.17.0',
    'absl-py',
    'rouge-score',
    'pandas'
]

extra_dev = [
    *extra_t5,
    *extra_examples,
]

setup(
    name='py-sled',
    version='0.1.7',

    python_requires='>=3.7.0',
    install_requires=[
        'transformers>=4.21.0',
        'makefun>=1.14.0',
    ],
    extras_require={
        't5': extra_t5,
        'examples': extra_examples,
        'dev': extra_dev
    },
    description='SLED models use pretrained, short-range encoder-decoder models, and apply them over long-text inputs '\
                'by splitting the input into multiple overlapping chunks, encoding each independently and '\
                'perform fusion-in-decoder',

    url='https://github.com/Mivg/SLED',
    author='Maor Ivgi',
    author_email='maor.ivgi@cs.tau.ac.il',

    packages=find_packages(exclude=("tests*", "examples*")),  # also added to manifest.in due to https://stackoverflow.com/a/46320848

    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing'
    ],
)
# build with `python setup.py sdist`
# upload with `python3 -m  twine upload dist/<the package>`