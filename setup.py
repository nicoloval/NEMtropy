from setuptools import setup


with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding='utf-8') as history_file:
    history = history_file.read()


setup(
    name="netrec",
    author="Nicolo' Vallarano, Emiliano Marchese",
    author_email='nicolo.vallarano@imtlucca.it, emiliano.marchese@imtlucca.it',
    packages=["netrecon"],
    package_dir={'': 'src'},
    version="0.1.0",
    description="bla",
    license="GNU General Public License v3",
    install_requires=["numpy==1.19.5",
                      "numba==0.52",
                      "networkx==2.5",
                      "scipy==1.6.0",
                      ],
    extras_require={
        "dev": ["pytest==6.0.1",
                "flake8==3.8.3",
                "wheel==0.35.1",
                "matplotlib==3.3.2",
                "check-manifest==0.44",
                "setuptools==47.1.0",],
        },
)
