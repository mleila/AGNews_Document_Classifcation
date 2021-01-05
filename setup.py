import os
from distutils.core import setup

scripts = [os.path.join('bin', f) for f in os.listdir('./bin')]
packages = ['news_classifier']
setup(name='news_classifier', version='0.1', scripts=scripts, packages=packages)
