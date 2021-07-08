---
# Installation
---

[TOC]

Building from Source
--------------------

Building Meep directly from the source code can be challenging for users unfamiliar with building Unix software. This is mainly because of the numerous prerequisites that must be installed as well as the need to specify in the build scripts where these packages are to be found.

Meep's build systems uses the standard [GNU Autotools](https://en.wikipedia.org/wiki/GNU_Build_System) `./configure && make && make install` machinery, but requires a number of prerequisites in order to obtain a full-featured Meep installation: [MPB](http://mpb.readthedocs.io/en/latest/), [Libctl](https://github.com/NanoComp/libctl), [Harminv](https://github.com/NanoComp/harminv), [libGDSII](https://github.com/HomerReid/libGDSII), [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), [OpenMP](https://en.wikipedia.org/wiki/OpenMP), [HDF5](https://support.hdfgroup.org/HDF5/), [Python](https://www.python.org/), and [Guile](https://www.gnu.org/software/guile/). MPB and Harminv, in turn, require [LAPACK and BLAS](http://www.netlib.org/lapack/lug/node11.html) and [FFTW](http://fftw.org/) to be installed.

Gzipped tarballs of stable versions of the source are available on the [releases page](https://github.com/NanoComp/meep/releases), and you can also do a `git clone` of the master branch of the [Meep repository on Github](https://github.com/NanoComp/meep) if you have Autotools installed. For more information, see [Build From Source](Build_From_Source.md).

Conda Packages
---------------

### Official Releases

The **recommended** way to install PyMeep is using the [Conda](https://conda.io/docs/) package manager. The precompiled binaries run as *fast or faster* than the typical build from source, are simple to install, can be upgraded easily, and take advantage of newer compilers and dependencies than those available in typical systems (e.g., [gcc](https://en.wikipedia.org/wiki/GNU_Compiler_Collection) 7.3.0 vs. Ubuntu 16.04's 5.4; a gap of nearly three years of advances in compiler optimization). Obviously, building from source can still provide advantages if you have access to special hardware or performance libraries that require specific compiler flags (e.g., [icc](https://en.wikipedia.org/wiki/Intel_C%2B%2B_Compiler)); building from source is also required if you are interested in working on the Meep [source code](https://github.com/NanoComp/meep), are performing system-wide installations on a server, or are using systems unsupported by Conda (e.g., supercomputers with Cray MPI).

Binary packages for serial and parallel PyMeep on Linux and macOS are currently available (64 bit architectures only), and are [updated with each new Meep release](https://github.com/conda-forge/pymeep-feedstock). Note: the Conda packages will *not* work on native [Windows](#installation-on-windows) (unless you install the Ubuntu terminal app) and do *not* include the Scheme interface which must be [built from source](Build_From_Source.md). The easiest way to get started is to install [Miniconda](https://conda.io/miniconda.html), which comes with everything necessary to create Python environments with Conda. For example, to install Miniconda with Python 3 on Linux:

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p <desired_prefix>
export PATH=<desired_prefix>/bin:$PATH
```

Next, we create a Conda environment for PyMeep to isolate it from other Python libraries that may be installed.

```bash
conda create -n mp -c conda-forge pymeep
```

This creates an environment called "mp" (you can name this anything you like) with PyMeep and all its dependencies. This will default to the version of Python in your Miniconda installation (Python 3 for us since we installed Miniconda3), but if you want to work with Python 2, just add `python=2` to the end of the command.

Next, we need to activate the environment before we can start using it.

```bash
conda activate mp
```

Now, `python -c 'import meep'` (or `python3 -c 'import meep'`) should work, and you can try running some of the examples in [meep/python/examples](https://github.com/NanoComp/meep/tree/master/python/examples).

**Note:** There is currently an issue with openblas 0.3.5 that causes segmentation faults on newer Skylake X-series cpus. If `import meep` results in an "illegal instruction" error, downgrade openblas to version `0.3.4` as follows:

```bash
conda install -c conda-forge openblas=0.3.4
```

Warning: The `pymeep` package is built to work with OpenBLAS, which means numpy should also use OpenBLAS. Since the default numpy is built with MKL, installing other packages into the environment may cause conda to switch to an MKL-based numpy. This can cause segmentation faults when calling MPB. To work around this, you can make sure the `no-mkl` conda package is installed, make sure you're getting packages from the `conda-forge` channel (they use OpenBLAS for everything), or as a last resort, run `import meep` before importing any other library that is linked to MKL. When installing additional packages into the `meep` environment, you should always try to install using the `-c conda-forge` flag. `conda` can occasionally be too eager in updating packages to new versions which can leave the environment unstable. If running `conda install -c conda-forge <some-package>` attempts to replace `conda-forge` packages with equivalent versions from the `defaults` channel, you can force it to only use channels you specify (i.e., arguments to the `-c` flag) with the `--override-channels` flag.

Installing parallel PyMeep follows the same pattern, but the package "build string" must be specified to bring in the MPI variant:

```bash
conda create -n pmp -c conda-forge pymeep=*=mpi_mpich_*
conda activate pmp
```
The first `*` requests the latest version of Pymeep, and the `mpi_mpich_*` says to get a version that includes "mpi_mpich" in the build string (the packages are currently built with the MPICH implementation of MPI).

The environment includes `mpi4py`, so you can run an MPI job with 4 processes like this:

```bash
mpirun -np 4 python <script_name>.py
```

If you run into issues, make sure your `PYTHONPATH` environment variable is unset.

**Note:** If you experience crashes when using `matplotlib` on macOS, try importing `meep` before importing `matplotlib`. In addition add the following line to your `~/.matplotlib/matplotlibrc` file to force the `TkAgg` backend:
```
backend: TkAgg
```

**Note:** For pymeep-parallel on macOS, a [bug](https://github.com/open-mpi/ompi/issues/2956) in openmpi requires that the environment variable `TMPDIR` be set to a short path like `/tmp`. Without this workaround, you may see errors similar to this:

```bash
[laptop:68818] [[53415,0],0] ORTE_ERROR_LOG: Bad
parameter in file ../../orte/orted/pmix/pmix_server.c at line 264

[laptop:68818] [[53415,0],0] ORTE_ERROR_LOG: Bad
parameter in file ../../../../../orte/mca/ess/hnp/ess_hnp_module.c at line
666
```

**Note:** To update, `pymeep`, you can do `conda update -c conda-forge pymeep`.  If you run into problems (e.g. some other update has interfered with your environment), you can instead create a new environment from scratch each time.

#### Older Releases

Older releases of PyMeep are available on the `conda-forge` channel. The full list of available versions is [here](https://anaconda.org/conda-forge/pymeep/files). Examples:

```bash
# Create an environment with the serial version of pymeep 1.8.0
conda create -n mp1.8 -c conda-forge pymeep=1.8.0
# Create an environment with the parallel version of pymeep 1.9.0
conda create -n pmp1.9 -c conda-forge pymeep=1.9.0=mpi_mpich_*
```

Note that parallel (MPI) versions are only available with `pymeep >= 1.8.0`.

### Nightly Builds

To experiment with new features before they are distributed in an official release, you can try the [nightly-development builds](https://github.com/Simpetus/pymeep-nightly-recipe). They are hosted on the `simpetus` channel. Currently, the nightly builds are only available for Python 2.7 and 3.6.

```bash
# Serial pymeep
conda create -n mp_test -c simpetus -c conda-forge pymeep
# Parallel pymeep
conda create -n pmp_test -c simpetus -c conda-forge pymeep=*=mpi_mpich*
```

### Version Number

You can determine the version number as well as the most recent commit of the Meep module via:

```py
import meep as mp
print(mp.__version__)
```

This will show something like `1.11.0-1-g415bc8eb` where the first three digits (`1.11.0`) refer to a stable tarball release, the following digit is the number of commits after this stable release, and the eight characters following the `g` in the final string refer to the commit hash.

### Non-Networked Systems

To install the PyMeep Conda package on a [non-networked system](https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/#installing-packages-on-a-non-networked-air-gapped-computer), using the bz2 tarball of the [official release](https://anaconda.org/conda-forge/pymeep/files) or [nightly build](https://anaconda.org/simpetus/pymeep/files) will *not* work without the dependencies. A possible workaround is [Conda-Pack](https://github.com/conda/conda-pack).

Installation on Linux
-------------------------

For most [Linux distributions](https://en.wikipedia.org/wiki/Linux_distribution), there should be precompiled packages for most of Meep's prerequisites below, and we *highly* recommend installing those prerequisites using the available packages for your system whenever possible. Using precompiled packages means that you don't have to worry about how to install things manually. You are using packages which have already been tweaked to work well with your system, and usually your packages will be automatically upgraded when you upgrade the rest of your system.

There are upcoming packages for [Meep version 1.17.1](https://github.com/NanoComp/meep/releases/tag/v1.17.1) with Python interface for Ubuntu 21.04 ("Hirsute Hippo") ([serial](https://packages.ubuntu.com/hirsute/python3-meep) and [parallel](https://packages.ubuntu.com/hirsute/python3-meep-openmpi)) and Debian 11 ("Bullseye") ([serial](https://packages.debian.org/bullseye/python3-meep) and [parallel](https://packages.debian.org/bullseye/python3-meep-openmpi)) which will be released in April 2021.

In the meantime, the following dependencies are already available as precompiled packages: BLAS and LAPACK possibly as part of a package for [Atlas BLAS](https://en.wikipedia.org/wiki/Automatically_Tuned_Linear_Algebra_Software), Guile, MPI, and HDF5. One thing to be careful of is that many distributions split packages into two parts: one main package for the libraries and programs, and a **devel** package for [header files](https://en.wikipedia.org/wiki/Header_file) and other things needed to compile software using those libraries. You will need to install **both**. So, for example, you will probably need both a `guile` package (probably installed by default) and a `guile-dev` or `guile-devel` package (probably *not* installed by default), and similarly for HDF5 etcetera. You will probably also want to install a `libpng-dev` or `libpng-devel` package in order to compile the `h5topng` utility in [h5utils](https://github.com/NanoComp/h5utils/blob/master/README.md).

Installation on macOS 
-----------------------

Since [macOS](https://en.wikipedia.org/wiki/macOS) is, at its heart, a Unix system, one can, in principle compile and install Meep and all its prerequisites just as on any other Unix system. However, this process is much easier using the [Homebrew](https://en.wikipedia.org/wiki/Homebrew_(package_management_software)) package to install most of the prerequisites, since it will handle dependencies and other details for you. You will need [administrator privileges](http://support.apple.com/kb/PH3920) on your Mac.

The first steps are:

-   Install [Xcode](https://en.wikipedia.org/wiki/Xcode), the development/compiler package from Apple, free from the [Apple Xcode web page](https://developer.apple.com/xcode/).
-   Install Homebrew: download from the [Homebrew site](http://brew.sh/) and follow the instructions there.
-   Run the following commands in the terminal to compile and install the prerequisites. This may take a while to complete because it will install lots of other stuff first

```sh
brew doctor
brew install hdf5 
brew install openblas 
brew install guile
brew install fftw
```


Now, install the Harminv, libctl, MPB, and Meep packages from source. Download [Harminv](https://github.com/NanoComp/harminv/blob/master/README.md) and, in the `harminv` directory, do:

```sh
./configure && make && make install
```

Use the same commands for [libctl](https://libctl.readthedocs.io), [MPB](https://mpb.readthedocs.io), and Meep. For more detailed information, see [Build From Source](Build_From_Source.md).

Download and install H5utilds: https://github.com/NanoComp/h5utils/blob/master/README.md

You are done, and can now run Meep (Scheme interface) just by typing `meep`. You can run `make check` in the meep directory if you want to perform a self-test.

To build the latest version of Meep from source on macOS Sierra, follow these [instructions](https://www.mail-archive.com/meep-discuss@ab-initio.mit.edu/msg05811.html).

Installation on Windows
----------------------------

Native Windows installation is currently unsupported. The recommended procedure is to install Ubuntu using the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10). This gives you access to a bash terminal running Ubuntu from within Windows. From there you can install the Conda packages as described above. The drawback is that you can't see plots from matplotlib (though saving them to disk and opening them from Windows works fine). The easiest way around this is to add the `jupyter` package to the `conda create ...` command. This will allow you to run a [Jupyter notebook](https://jupyter.readthedocs.io/en/latest/) in the browser, and from there you can visualize plots interactively.
