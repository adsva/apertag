from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("perceptron", ["perceptron.pyx"])]

with open('README.rst') as f:
    readme = f.read()

setup(
    name='Apertag',
    author = "Adam Svanberg",
    author_email = "asvanberg@gmail.com",
    version='1.0.2',
    py_modules=['apertag'],
    description='Averaged Perceptron Sequence Tagger',
    url='https://github.com/adsva/apertag',
    long_description=readme,
    classifiers=[
        'Programming Language :: Python',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',

    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[numpy.get_include()],
)
