import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='peslearn',
        version="0.1.1",
        description='Automated Construction of Machine Learning Models of Molecular Potential Energy Surfaces.',
        author='Adam Abbott',
        author_email='adabbott@uga.edu',
        url="https://github.com/CCQC/PES-Learn",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7','GPy>=1.9','scikit-learn>=0.20','pandas>=0.24','hyperopt>=0.1.1','cclib>=1.6', 'torch>=1.0.1'
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3', 
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest-cov',
            ],
        },

        tests_require=[
            'pytest-cov',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
    )
