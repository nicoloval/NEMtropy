from setuptools import setup, find_packages


with open('README.md', "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="NEMtropy",
    author="Nicolo' Vallarano, Emiliano Marchese",
    author_email='nicolo.vallarano@imtlucca.it, emiliano.marchese@imtlucca.it',
    packages=["NEMtropy"],
    package_dir={'': 'src'},
    version="1.0.6",
    description="NEMtropy is a Maximum-Entropy toolbox for networks, it"
                " provides the user with a state of the art solver for a range variety"
                " of Maximum Entropy Networks models derived from the ERGM family."
                " This module allows you to solve the desired model and generate a"
                " number of randomized graphs from the original one:"
                " the so-called graphs ensemble.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GNU General Public License v3",
    url="https://github.com/nicoloval/NEMtropy/",
    download_url="https://github.com/nicoloval/NEMtropy/archive/v1.0.6.zip",
    keywords=['Network reconstruction', 'Networks Null Models',
              'Maximum Entrophy Methods'],
    classifiers=[
                'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',
                'Programming Language :: Python :: 3.8',
                ],
    install_requires=[
                      "numpy>=1.17",
                      "scipy>=1.4",
                      "networkx>=2.4",
                      "powerlaw>=1.4"
                      ],
    extras_require={
        "dev": [
                "pytest>=6.0.1",
                "flake8>=3.8.3",
                "wheel>=0.35.1",
                "check-manifest>=0.44",
                "setuptools>=47.1.0",
                "twine>=3.2.0",
                "tox>=3.20.1",
                "powerlaw>=1.4.4",
                ],
        },
)
