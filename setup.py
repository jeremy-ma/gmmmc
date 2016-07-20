from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os

sourcefiles = [os.getcwd() + '/gmmmc/fastgmm/' + 'fast_likelihood_threaded.cpp']
ext_modules = [Extension("fastgmm",
                          sourcefiles,
                          include_dirs = [np.get_include()],
                          extra_compile_args=['-O3', '-fopenmp', '-lc++'],
                          extra_link_args=['-fopenmp'],
                          language='c++')]

setup(
    name='gmmmc',
    version='0.2',
    packages=['gmmmc', 'gmmmc.tests', 'gmmmc.priors', 'gmmmc.fastgmm', 'gmmmc.proposals'],
    url='',
    license='',
    author='Jeremy Ma',
    author_email='jeremy.ma@student.unsw.edu.au',
    description='Functions for drawing Monte Carlo samples from GMM parameter space',
    ext_modules=ext_modules
)
