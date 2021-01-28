from setuptools import setup


with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst', encoding='utf-8') as history_file:
    history = history_file.read()


setup(
    name="netrecon",
    author="Nicolo' Vallarano, Emiliano Marchese",
    author_email='nicolo.vallarano@imtlucca.it, emiliano.marchese@imtlucca.it',
    packages=["netrecon"],
    package_dir={'': 'src'},
    version="0.1.0",
    description="bla",
    license="GNU General Public License v3",
    url = "https://github.com/nicoloval/classes/",
    download_url = "https://github.com/nicoloval/classes/archive/master.zip",
    keywords = ['Network reconstruction', 'Networks Null Models', 'Maximum Entrophy Methods'],
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Network Science Community',      # Define that your audience are developers
    'Topic :: ',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.8',
                ],
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
                "setuptools==47.1.0",
                "twine==3.2.0",
                "tox==3.20.1",
                "powerlaw==1.4.4",
                ],
        },
)
