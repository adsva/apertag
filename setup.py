from distutils.core import setup

with open('README.rst') as f:
    readme = f.read()

setup(
    name='Apertag',
    author = "Adam Svanberg",
    author_email = "asvanberg@gmail.com",
    version='1.0.1',
    py_modules=['apertag'],
    description='Averaged Perceptron Sequence Tagger',
    url='https://github.com/adsva/apertag',
    long_description=readme,
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
