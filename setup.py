from itertools import chain

from setuptools import find_packages, setup

from checkers import __author__, __author_email__, __description__, __name__

# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
# fmt: off
extra_feature_requirements = {
    "doc": [
        "ipykernel",  # Used by nbsphinx to execute notebooks
        "nbsphinx                       >= 0.7",
        "numpydoc",
        "sphinx                         >= 3.0.2",
        "sphinx-codeautolink[ipython]",
        "sphinx-copybutton              >= 0.2.5",
        "sphinx-design",
        "sphinx-gallery                 < 0.11",
        "sphinx-last-updated-by-git",
        "pydata-sphinx-theme            >= 0.13.1",
        "sphinxcontrib-bibtex           >= 1.0",
        "scikit-image",
        "scikit-learn",
    ],
    "tests": [
        "coverage                       >= 5.0",
        "numpydoc",
        "pytest                         >= 5.4",
        "pytest-cov                     >= 2.8.1",
        "pytest-xdist",
    ],
}
extra_feature_requirements["dev"] = [
    "black[jupyter]",
    "isort                              >= 5.10",
    "manifix",
    "outdated",
    "packaging",
    "pre-commit                         >= 1.16",
] + list(chain(*list(extra_feature_requirements.values())))
# fmt: on

# Remove the "raw" ReStructuredText directive from the README so we can
# use it as the long_description on PyPI
readme = open("README.rst").read()
readme_split = readme.split("\n")
for i, line in enumerate(readme_split):
    if line == ".. EXCLUDE":
        break
long_description = "\n".join(readme_split[i + 2 :])

setup(
    name=__name__,
    license="GPLv3",
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/x-rst",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    packages=find_packages(exclude=["checkers/tests"]),
    extras_require=extra_feature_requirements,
    # fmt: off
    install_requires=[
#        "dask[array]",
#        "diffpy.structure       >= 3.0.2",
#        "h5py",
#        "matplotlib             >= 3.3",
#        "matplotlib-scalebar",
#        "numba",
        "numpy",
#        "numpy-quaternion",
        "pooch                  >= 0.13",
        "scipy",
#        "tqdm",
    ],
    # fmt: on
    package_data={"": ["LICENSE", "README.rst", "readthedocs.yaml"], "checkers": ["*.py"]},
)
