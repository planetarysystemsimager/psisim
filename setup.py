from setuptools import setup, find_packages

setup(
    name='psisim',
    version='0.1',
    description='PSISIM: a simulator for TMT-PSI',
    url='TBD',
    author='Max+Jason+Others',
    license='BSD',
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        ],
    install_requires=['numpy', 'scipy', 'astropy']
    )
