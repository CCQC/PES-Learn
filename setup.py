import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='PES-Learn',
        version="0.0.1",
        description='Automated Construction of Machine Learning Models of Molecular Potential Energy Surfaces.',
        author='Adam Abbott',
        author_email='adabbott@uga.edu',
        url="https://github.com/adabbott/PES-Learn",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7',
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
                'tox',
            ],
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
            'tox',
        ],

        classifiers=[
            'Development Status :: 1 - Planning',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
    )
