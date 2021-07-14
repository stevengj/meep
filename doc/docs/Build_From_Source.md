---
# Build From Source
---

The main effort in installing Meep lies in installing the various dependency packages. This requires some understanding of how to install software on Unix systems.

It is also possible to install Meep on Windows systems. For Windows 10, you can install the [Ubuntu 16.04](https://www.microsoft.com/en-us/p/ubuntu-1604-lts/9pjn388hp8c9) or [18.04](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6) terminal as an app (via the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about) framework) and then follow the instructions for [obtaining the Conda packages](Installation.md#conda-packages) (recommended) or [building from source](Build_From_Source.md#building-from-source). For Windows 8 and older versions, you can use the free Unix-compatibility environment [Cygwin](http://www.cygwin.org/) following these [instructions](http://novelresearch.weebly.com/installing-meep-in-windows-8-via-cygwin.html).

For those installing Meep on a supercomputer, a note of caution: most supercomputers have multiple compilers installed, and different versions of libraries compiled with different compilers. Meep is written in C++, and it is almost impossible to mix C++ code compiled by different compilers &mdash; pick one set of compilers by one vendor and stick with it consistently.

[TOC]

Unix Installation Basics
------------------------

### Installation Paths

First, let's review some important information about installing software on Unix systems, especially in regards to installing software in non-standard locations. None of these issues are specific to Meep, but they've caused a lot of confusion among users.

Most of the software below, including Meep, installs under `/usr/local` by default. That is, libraries go in `/usr/local/lib`, programs in `/usr/local/bin`, etc. If you don't have `root` privileges on your machine, you may need to install somewhere else, e.g. under `$HOME/install` (the `install/` subdirectory of your home directory). Most of the programs below use a GNU-style `configure` script, which means that all you would do to install there would be:

```sh
 ./configure --prefix=$HOME/install
```

when configuring the program. The directories `$HOME/install/lib` etc. are created automatically as needed.

#### Paths for Configuring

There are two further complications. First, if you install dependencies in a non-standard location like `$HOME/install/lib`, you will need to tell the compilers where to find the libraries and header files that you installed. You do this by passing two variables to `./configure`:

```bash
./configure LDFLAGS="-L$HOME/install/lib" CPPFLAGS="-I$HOME/install/include"   ...other flags...
```

Of course, substitute whatever installation directory you used. You may need to include multiple `-L` and `-I` flags separated by spaces if your machine has stuff installed in several non-standard locations.

You might also need to update your `PATH` so that you can run the executables; e.g. if we installed in our home directory as described above, we would do:

```bash
export PATH="$HOME/install/bin:$PATH"
```

#### Paths for Running (Shared Libraries)

Second, many of the packages installed below (e.g. Guile) are installed as shared libraries. You need to make sure that your runtime linker knows where to find these shared libraries. The bad news is that every operating system does this in a slightly different way.
If you installed all of your libraries in a standard location on your operating system (e.g. `/usr/lib`), then the runtime linker will look there already and you don't need to do anything.  Otherwise, if you compile things like `libctl` and install them into a "nonstandard" location (e.g. in your home directory), you will need to tell the runtime linker where to find them.

There are several ways to do this.  Suppose that you installed libraries into the directory `$HOME/install/lib`.   The most robust option is probably to include this path in the linker flags:

```bash
./configure LDFLAGS="-L$HOME/install/lib -Wl,-rpath,$HOME/install/lib" ...other flags...
```

There are also some other ways.  If you use Linux, have superuser privileges, and are installing in a system-wide location (not your home directory), you can add the library directory to `/etc/ld.so.conf` and run `/sbin/ldconfig`.

On many systems, you can also specify directories to the runtime linker via the `LD_LIBRARY_PATH` environment variable. In particular, by `export LD_LIBRARY_PATH="$HOME/install/lib:$LD_LIBRARY_PATH"`; you can add this to your `.profile` file (depending on your shell) to make it run every time you run your shell. On MacOS, a security feature called [System Integrity Protection](https://en.wikipedia.org/wiki/System_Integrity_Protection) causes the value of `LD_LIBRARY_PATH` to be ignored, so using environment variables won't work there.

### Fun with Fortran

Meep, along with many of the libraries it calls, is written in C or C++, but it also calls libraries such as BLAS and LAPACK (see below) that are usually compiled from Fortran. This can cause some added difficulty because of the various linking schemes used by Fortran compilers. The `configure` script attempts to detect the Fortran linking scheme automatically, but in order for this to work you must use the same Fortran compiler and options with Meep as were used to compile BLAS/LAPACK.

By default, Meep looks for a vendor Fortran compiler first (`f77`, `xlf`, etcetera) and then looks for GNU `g77`. In order to manually specify a Fortran compiler `foobar` you would configure it with `./configure F77=foobar ...`.

If, when you compiled BLAS/LAPACK, you used compiler options that alter the linking scheme (e.g. `g77`'s `-fcase-upper` or `-fno-underscoring`), you will need to pass the same flags to Meep via `./configure FFLAGS=...flags... ...`.

### Picking a Compiler

It is often important to be consistent about which compiler you employ. This is especially true for C++ software. To specify a particular C compiler `foo`, configure with `./configure CC=foo`; to specify a particular C++ compiler `foo++`, configure with `./configure CXX=foo++`; to specify a particular Fortran compiler `foo90`, configure with `./configure F77=foo90`.

### Linux and BSD Binary Packages

If you are installing on your personal Linux or BSD machine, then precompiled binary packages are likely to be available for many of these packages, and may even have been included with your system. On Debian systems, the packages are in `.deb` format and the built-in `apt-get` program can fetch them from a central repository. On Red Hat, SuSE, and most other Linux-based systems, binary packages are in RPM format.  OpenBSD has its "ports" system, and so on.

**Do not compile something from source if an official binary package is available.**  For one thing, you're just creating pain for yourself.  Worse, the binary package may already be installed, in which case installing a different version from source will just cause trouble.

One thing to watch out for is that libraries like LAPACK, Guile, HDF5, etcetera, will often come split up into two or more packages: e.g. a `guile` package and a `guile-devel` package. You need to install **both** of these to compile software using the library.

Required Dependencies
---------------------

### libctl

[libctl](https://libctl.readthedocs.io) is required to use the Python or Scheme interfaces. If you don't install it, you can only use the C++ interface. libctl version **4.0 or later** is required.

If you only want the Python interface, it is possible to install the library portions of libctl without having Scheme/Guile by configuring libctl with `--without-guile`.   You should then configure MPB `--without-libctl`, which builds only the C library of MPB, but still configure Meep `--with-libctl`: Meep will then build the Python interface for both Meep and MPB using the libctl and MPB libraries.

Instead of using Guile directly in our Scheme interface, we separated much of the user interface code into a package called libctl, in the hope that this might be more generally useful. libctl automatically handles the communication between the program and Guile, converting complicated data structures and so on, to make it even easier to use Guile to control scientific applications. Download libctl from the [libctl page](https://libctl.readthedocs.io), unpack it, and run the usual `configure`, `make`, `make install` sequence. You'll also want to browse the [libctl manual](https://libctl.readthedocs.io), as this will give you a general overview of what the user interface will be like.

If you are not the system administrator of your machine, and/or want to install libctl somewhere else like your home directory, you can do so with the standard `--prefix=dir` option to `configure`. The default prefix is `/usr/local`. In this case, however, you'll need to specify the location of the libctl shared files for the Meep package, using the `--with-libctl=dir/share/libctl` option to our `configure` script.

### Python

If you have Python on your system, then the Meep compilation scripts automatically build and install the `meep` Python module, which works with both the serial and parallel (MPI) versions of Meep. Note: Meep's [visualization module](Python_User_Interface.md#data-visualization) includes animation routines which require [matplotlib](https://matplotlib.org/) version `3.1`+ and the [adjoint solver](Python_Tutorials/Adjoint_Solver.md) requires [autograd](https://github.com/HIPS/autograd).

By default, Meep's Python module is installed for the program `python` on your system.  If you want to install using a different Python program, e.g. `python3`, pass `PYTHON=python3` (or similar) to the Meep `configure` script. An Anaconda (`conda`) [package for Meep](Installation.md#conda-packages) is also available on some systems.

Optional Dependencies
---------------------

### BLAS and LAPACK

BLAS and LAPACK libraries are required in order to install [Harminv](https://github.com/NanoComp/harminv/blob/master/README.md). Harminv is not *required* for Meep, but is strongly recommended for use in resonant-mode computation.

Note also that Meep's usage of BLAS/LAPACK, via Harminv, is not generally performance critical. So, it doesn't matter too much whether you install an especially optimized BLAS library. However, it makes a big difference if you also use [MPB](https://mpb.readthedocs.io).

### BLAS

The first thing you must have on your system is a BLAS implementation. "BLAS" stands for "Basic Linear Algebra Subroutines," and is a standard interface for operations like matrix multiplication. It is designed as a building-block for other linear-algebra applications, and is used directly by LAPACK (see below). By using it, we can take advantage of many highly-optimized implementations of these operations that have been written to the BLAS interface. Note that you will need implementations of BLAS levels 1-3.

You can find more BLAS information, as well as a basic implementation, on its [homepage](http://www.netlib.org/blas/). Once you get things working with the basic BLAS implementation, it might be a good idea to try and find a more optimized BLAS code for your hardware. Vendor-optimized BLAS implementations are available as part of the Intel MKL, HP CXML, IBM ESSL, SGI sgimath, and other libraries. An excellent, high-performance, free-software BLAS implementation is  [OpenBLAS](http://www.openblas.net). Another is [ATLAS](http://math-atlas.sourceforge.net/).

Note that the generic BLAS does not come with a `Makefile`; compile it with something like: </nowiki>

```sh
  wget http://www.netlib.org/blas/blas.tgz
  gunzip blas.tgz
  tar xf blas.tar
  cd BLAS
  f77 -c -O3 *.f   # compile all of the .f files to produce .o files
  ar rv libblas.a *.o    #  combine the .o files into a library
  su -c "cp libblas.a /usr/local/lib"   # switch to root and install
```

Replace `-O3` with your favorite optimization options. On Linux, this could be `g77 -O3 -fomit-frame-pointer -funroll-loops -malign-double`. Note that MPB looks for the standard BLAS library with `-lblas`, so the library file should be called `libblas.a` and reside in a standard directory like `/usr/local/lib`. See also below for the `--with-blas=lib` option to MPB's `configure` script, to manually specify a library location.

### LAPACK

LAPACK, the Linear Algebra PACKage, is a standard collection of routines, built on top of BLAS, for more-complicated (dense) linear algebra operations like matrix inversion and diagonalization. You can download LAPACK from its [homepage](http://www.netlib.org/lapack).

Note that Meep looks for LAPACK by linking with `-llapack`. This means that the library must be called `liblapack.a` and be installed in a standard directory like `/usr/local/lib`. Alternatively, you can specify another directory via the `LDFLAGS` environment variable as described earlier. See also below for the `--with-lapack=lib` option to our `configure` script, to manually specify a library location.

We currently recommend installing OpenBLAS which includes LAPACK so you do not need to install it separately.

### Harminv

To use Meep to extract resonant frequencies and decay rates, you must install [Harminv](https://github.com/NanoComp/harminv/blob/master/README.md) which requires BLAS and LAPACK.

See the [Harminv installation](https://github.com/NanoComp/harminv/blob/master/doc/installation.md) instructions.

### MPI

Optionally, Meep is able to run on a distributed-memory parallel machine, and to do this we use the standard message-passing interface (MPI). Most supercomputers already have an MPI implementation installed. The recommended implementation is [Open MPI](http://www.open-mpi.org/). MPI is **not required** to compile the serial version of Meep.

In order for the MPI version of the Python and Scheme interface to run successfully, we have a slightly nonstandard requirement: each process must be able to read from the disk. This way, Python and Guile can boot for each process and they can all read your simulation file in parallel. Most supercomputers satisfy this requirement. On the other hand, the C++ interface to Meep does not have this requirement.

If you use Meep with MPI, you should compile HDF5 with MPI support as well (see [below](#hdf5)).

As described below, when you configure Meep with MPI support (`--with-mpi`), it installs itself as `meep` (for the Scheme interface), so it overwrites any serial installation. There is no need to have separate serial `meep` installed, however, because if you run the parallel Meep simply as `meep`, it runs on a single processor (to launch six processes involves `mpirun -np 6 meep`).

### HDF5

Meep outputs its fields and other volumetric data in the HDF5 format, so you must install the HDF5 libraries if you want to visualize the fields. [HDF](https://www.hdfgroup.org) is a widely-used, free, portable library and file format for multi-dimensional scientific data. There are two incompatible versions of HDF, HDF4 and HDF5 (no, not HDF1 and HDF2). We require the newer version, HDF5, which is supported by a number scientific of visualization tools, including [h5utils](https://github.com/NanoComp/h5utils/blob/master/README.md) utilities.

HDF5 supports parallel I/O under MPI which can be enabled by configuring it with `--enable-parallel`. You may also have to set the `CC` environment variable to `mpicc`. Unfortunately, the parallel HDF5 library then does not work with serial code, so you have may have to choose one or the other. We have some hacks in Meep to do parallel I/O even with the serial HDF5 library. These hacks work okay when you are using a small number of processors, but on large HPC clusters we strongly recommend using the parallel HDF5.

**Note:** If you have a version of HDF5 compiled with MPI parallel I/O support, then you need to use the MPI compilers to link to it, even when you are compiling the serial version of Meep.  Just use `./configure CC=mpicc CXX=mpic++` or whatever your MPI compilers are when configuring.

### libGDSII

[libGDSII](https://github.com/HomerReid/libGDSII) is a library for reading [GDSII](https://en.wikipedia.org/wiki/GDSII) binary data files. GDSII is a widely-used format for 2d/planar geometries supported by [electronic design automation](https://en.wikipedia.org/wiki/Electronic_design_automation) (EDA) circuit-layout editors (e.g., Cadence Virtuoso Layout, Silvaco Expert, KLayout, etc.) and semiconductor foundries.

### Guile

Guile is required in order to use the Scheme interface. If you don't install it, you can only use the C++ and/or Python interfaces.

Guile is an extension/scripting language implementation based on Scheme, and we use it to provide a rich, fully-programmable user interface with minimal effort. It's free, of course, and you can download it from the [Guile homepage](http://www.gnu.org/software/guile/). Guile is typically included with Linux systems.

- **Important:** Most Linux distributions come with Guile already installed. You can check by seeing whether you can run `guile --version` from the command line. In that case, do **not** install your own version of Guile from source &mdash; having two versions of Guile on the same system will cause problems. However, by default most distributions install only the Guile libraries and not the programming headers &mdash; to compile libctl and MPB, you should install the **guile-devel** or **guile-dev** package.

### MPB

The planewave expansion mode solver [MPB](https://mpb.readthedocs.io/en/latest/) is required for the [eigenmode source](Python_User_Interface.md#eigenmodesource) and [mode decomposition](Python_User_Interface.md#mode-decomposition) features.

**Note:** Meep can only link to the *serial* version of MPB.

Meep
----

Once you've installed all of the dependencies, you can install Meep via:

```sh
./configure
make
sudo make install
```

Assuming you've set your `LDFLAGS` etcetera, the configure script should find all of the libraries you've installed and, with luck, compile successfully. The `sudo` in the last command uses administrator privileges to install the binaries in standard system directories. Alternatively, you can just use `make install` if you have used `--prefix` to change the installation directory to something like your home directory. This is described below. To make sure Meep is working, you can run its test suite via:

```sh
make check
```

When running the test suite, the number of MPI processes is two and if OpenMP is enabled (as described further below) the number of threads is two. These default values can be overridden by passing `RUNCODE` to `make`, e.g.:

```sh
make RUNCODE="env OMP_NUM_THREADS=4 mpirun -np 3" check
```

If [`Coverage.py`](https://coverage.readthedocs.io/en/coverage-5.5/index.html) is found on your machine, the Python test suite is always run in serial mode (i.e., no MPI).

Note: several of the unit tests generate output files which are written to disk. The C++ test suite in `meep/tests` outputs its files in the same subdirectory. The Python test suite in `meep/python/tests` outputs its files to a temporary system directory (i.e., /tmp, etc.).

The configure script accepts several flags to modify its behavior.

**`--prefix=dir`**
—
Install into `dir/bin`, etcetera, as described above.


**`--with-mpi`**
—
Attempt to compile a [parallel version of Meep](Parallel_Meep.md) using MPI; the resulting program will be installed as `meep` and can be run in either serial or parallel mode (the latter via `mpirun`). Requires MPI to be installed, as described above.  (You should install this *instead* of the serial Meep.) Note that the configure script attempts to automatically detect how to compile MPI programs, but this may fail if you have an unusual version of MPI or if you have several versions of MPI installed and you want to select a particular one. You can control the version of MPI selected by setting the `MPICXX` variable to the name of the compiler to use and the `MPILIBS` variable to any additional libraries that must be linked (e.g., `./configure MPICXX=foompiCC MPILIBS=-lfoo ...`).

**`--with-libctl=dir`**
—
If libctl was installed in a nonstandard location (i.e. neither `/usr` nor `/usr/local`), you need to specify the location of the libctl directory, *`dir`*. This is either `prefix/share/libctl`, where `prefix` is the installation prefix of libctl, or the original libctl source code directory. To configure *without* the libctl/Guile interface, use `--without-libctl`.

**`--without-python`**
—
Disable building the Python API for Meep.

**`--with-blas=lib`**
—
The `configure` script automatically attempts to detect accelerated BLAS libraries, like DXML (DEC/Alpha), SCSL and SGIMATH (SGI/MIPS), ESSL (IBM/PowerPC), ATLAS, and PHiPACK. You can, however, force a specific library name to try via `--with-blas=lib`.

**`--with-lapack=lib`**
—
Cause the `configure` script to look for a LAPACK library called *`lib`*. The default is to use `-llapack`.

**`--enable-debug`**
—
Compile for debugging, adding extra runtime checks and so on.

**`--enable-shared`**
—
Install the Meep libraries as shared libraries (i.e. dynamically linked) rather than as static libraries. This is off by default because shared libraries require the user to configure their runtime linker paths correctly (see "Paths for Running" above).

**`--without-hdf5`**
—
Install Meep without support for the HDF5 libraries (this means you won't be able to output fields and so on).

**`--enable-portable-binary`**
—
By default, Meep's `configure` script picks compiler flags to optimize Meep as much as possible for the machine you are compiling on. If you wish to run the *same compiled executable* on other machines, however, you need to tell it not to pick compiler flags that use features specific to your current processor. In this case you should pass `--enable-portable-binary` to `configure`. (This option is mainly useful for building binary packages for Debian, Fedora, etcetera.)

**`--with-gcc-arch=arch`, `--without-gcc-arch`**
—
By default, Meep's configure script tries to guess the gcc `-march` flag for the system you are compiling on using `-mtune` instead when `--enable-portable-binary` is specified. If it guesses wrong, or if you want to specify a different architecture, you can pass it here. Alternatively, if you want to compile using your native architecture, you can just use `--disable-portable-binary` and omit `--with-gcc-arch` entirely. This approach will leave it up to your compiler to automatically determine the architecture. If you want to omit `-march`/`-mtune` flags entirely, pass `--without-gcc-arch`.

**`--with-openmp`**
—
This flag enables some experimental support for [OpenMP](https://en.wikipedia.org/wiki/OpenMP) multithreading parallelism on multi-core machines (*instead* of MPI, or in addition to MPI if you have multiple processor cores per MPI process). Currently, only multi-frequency [`near2far`](Python_User_Interface.md#near-to-far-field-spectra) calculations are sped up this way, but in the future this [may be expanded](https://github.com/NanoComp/meep/issues/228) with additional OpenMP parallelism. When you run Meep, you can first set the `OMP_NUM_THREADS` environment variable to the number of threads you want OpenMP to use.

### Floating-Point Precision of the Fields and Materials Arrays

By default, the C/C++ arrays used in Meep to store the time-domain fields ($\mathbf{E}$, $\mathbf{D}$, $\mathbf{H}$) and materials ($\varepsilon$) are defined using [double-precision floating point](https://en.wikipedia.org/wiki/Double-precision_floating-point_format). Updating the fields arrays generally dominates the computational cost of the simulation because it occurs at every voxel in the cell and at every timestep. Because [discretization errors](https://en.wikipedia.org/wiki/Discretization_error) which include the discontinuous material interfaces (as described in [Subpixel Smoothing](Subpixel_Smoothing.md)) as well as the [numerical dispersion](https://en.wikipedia.org/wiki/Numerical_dispersion) of the Yee grid typically dominates the [floating-point roundoff error](https://en.wikipedia.org/wiki/Round-off_error), the fields/materials arrays can be defined using [single-precision floating point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) to provide a significant speedup (by reducing the [memory bandwidth](https://en.wikipedia.org/wiki/Memory_bandwidth)) often without *any loss* in simulation accuracy.

This feature requires one to pass the `--enable-single` flag to
the Meep `configure` script before compiling.

As a demonstration of the potential improvement in runtime performance, for an experiment involving [computing the light-extraction efficiency of an OLED](http://www.simpetus.com/projects.html#meep_oled) which includes PMLs, DFT flux monitors, and Lorentzian susceptibilities, the timestepping rate (s/step) for the single-precision case using 20 MPI processes was ~50% that of double precision.

The DFT fields, however, are always defined using double-precision floating point. This is intended to mitigate the accumulation of round-off error for simulations with a large number of timesteps.

### Separating Build and Source Paths

Meep supports ["VPATH" builds](https://www.gnu.org/software/automake/manual/html_node/VPATH-Builds.html), where you compile in a separate directory from the source directory.  This is helpful if you want to keep the source directory in a pristine state, or if you want to build multiple binaries simultaneously from the same source tree.   Just create a build directory and execute the `configure` script by supplying its path, for example

```sh
mkdir meepbuild
cd meepbuild
/path/to/meep/configure ...options...
make
```

### Building From Source

The following instructions are for building parallel PyMeep with all optional features from source on Ubuntu 16.04/18.04. (There is a separate [script](http://ab-initio.mit.edu/~oskooi/meep_discuss/build_meep.sh) if you only want the Scheme interface.) The parallel version can still be run serially by running a script with just `python` instead of `mpirun -np 4 python`. If you really don't want to install MPI and parallel HDF5, just replace `libhdf5-openmpi-dev` with `libhdf5-dev`, and remove the `--with-mpi`, `CC=mpicc`, and `CPP=mpicxx` flags. The paths to HDF5 will also need to be adjusted to `/usr/lib/x86_64-linux-gnu/hdf5/serial` and `/usr/include/hdf5/serial`. Note that this script builds with Python 3 by default. If you want to use Python 2, just point the `PYTHON` variable to the appropriate interpreter when calling `autogen.sh` for building Meep, and use `pip` instead of `pip3`.

The entire build and install procedure can also be performed using an automated script:

```sh
mkdir -p /where/to/install/meep
cd /where/to/install/meep
wget https://raw.githubusercontent.com/NanoComp/meep/master/contrib/build-meep.sh
chmod +x build-meep.sh
./build-meep.sh
```

#### Ubuntu 16.04 and 18.04

There are a few differences in building for 16.04 and 18.04, so be sure to read the script and adjust appropriately.

```bash
#!/bin/bash

set -e

RPATH_FLAGS="-Wl,-rpath,/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi"
MY_LDFLAGS="-L/usr/local/lib -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi ${RPATH_FLAGS}"
MY_CPPFLAGS="-I/usr/local/include -I/usr/include/hdf5/openmpi"

sudo apt-get update

# If building on Ubuntu 18.04LTS, replace libpng16-dev with libpng-dev,
# and libpython3.5-dev with libpython3-dev.
sudo apt-get -y install     \
    build-essential         \
    gfortran                \
    libblas-dev             \
    liblapack-dev           \
    libgmp-dev              \
    swig                    \
    libgsl-dev              \
    autoconf                \
    pkg-config              \
    libpng16-dev            \
    git                     \
    guile-2.0-dev           \
    libfftw3-dev            \
    libhdf5-openmpi-dev     \
    hdf5-tools              \
    libpython3.5-dev        \
    python3-pip             \
    cmake                   \

mkdir -p ~/install

cd ~/install
git clone https://github.com/NanoComp/harminv.git
cd harminv/
sh autogen.sh --enable-shared
make && sudo make install

cd ~/install
git clone https://github.com/NanoComp/libctl.git
cd libctl/
sh autogen.sh --enable-shared
make && sudo make install

cd ~/install
git clone https://github.com/NanoComp/h5utils.git
cd h5utils/
sh autogen.sh CC=mpicc LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}"
make && sudo make install

cd ~/install
git clone https://github.com/NanoComp/mpb.git
cd mpb/
sh autogen.sh --enable-shared CC=mpicc LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}" --with-hermitian-eps
make && sudo make install

cd ~/install
git clone https://github.com/HomerReid/libGDSII.git
cd libGDSII/
sh autogen.sh
make && sudo make install

# The next line is only required on Ubuntu  16.04
sudo pip3 install --upgrade pip

pip3 install --user --no-cache-dir mpi4py
pip3 install --user Cython==0.29.16
export HDF5_MPI="ON"
pip3 install --user --no-binary=h5py h5py
pip3 install --user autograd
pip3 install --user scipy
pip3 install --user matplotlib>3.0.0
pip3 install --user ffmpeg

cd ~/install
git clone git://github.com/stevengj/nlopt.git
cd nlopt/
cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 && make && sudo make install

cd ~/install
git clone https://github.com/NanoComp/meep.git
cd meep/
sh autogen.sh --enable-shared --with-mpi --with-openmp PYTHON=python3 LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}"
make && sudo make install
```

You may want to add the following line to your `.profile` so Python can always find the `meep` (and `nlopt`) package:

```bash
# Ubuntu 16.04
export PYTHONPATH=/usr/local/lib/python3.5/site-packages:/usr/local/lib/python3/dist-packages
# Ubuntu 18.04
export PYTHONPATH=/usr/local/lib/python3.6/site-packages:/usr/local/lib/python3/dist-packages
```

#### CentOS 7

```bash
#!/bin/bash

set -e

RPATH_FLAGS="-Wl,-rpath,/usr/local/lib:/usr/local/lib/openmpi"
MY_LDFLAGS="-L/usr/local/lib -L/usr/local/lib/openmpi ${RPATH_FLAGS}"
MY_CPPFLAGS="-I/usr/local/include -I/usr/local/include/openmpi"

sudo yum -y --enablerepo=extras install epel-release
sudo yum -y install   \
    bison             \
    byacc             \
    cscope            \
    ctags             \
    cvs               \
    diffstat          \
    oxygen            \
    flex              \
    gcc               \
    gcc-c++           \
    gcc-gfortran      \
    gettext           \
    git               \
    indent            \
    intltool          \
    libtool           \
    patch             \
    patchutils        \
    rcs               \
    redhat-rpm-config \
    rpm-build         \
    subversion        \
    systemtap         \
    wget
sudo yum -y install    \
    openblas-devel     \
    fftw3-devel        \
    libpng-devel       \
    gsl-devel          \
    gmp-devel          \
    pcre-devel         \
    libtool-ltdl-devel \
    libunistring-devel \
    libffi-devel       \
    gc-devel           \
    zlib-devel         \
    openssl-devel      \
    sqlite-devel       \
    bzip2-devel        \
    ffmpeg

mkdir -p ~/install

cd ~/install
wget https://github.com/swig/swig/archive/rel-3.0.12.tar.gz
tar xvf rel-3.0.12.tar.gz
cd swig-rel-3.0.12
./autogen.sh
./configure
make -j
sudo make -j install

cd ~/install
wget https://ftp.gnu.org/gnu/guile/guile-2.0.11.tar.gz
tar xvf guile-2.0.11.tar.gz
cd guile-2.0.11
./configure
make -j
sudo make -j install

cd ~/install
wget https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz
tar xvf Python-3.6.5.tgz
cd Python-3.6.5
./configure --enable-optimizations
make -j
sudo make -j install

cd ~/install
wget https://download.open-mpi.org/release/open-mpi/v2.1/openmpi-2.1.1.tar.gz
tar xvf openmpi-2.1.1.tar.gz
cd openmpi-2.1.1/
./configure
make -j all
sudo make -j install

cd ~/install
git clone https://bitbucket.hdfgroup.org/scm/hdffv/hdf5.git
cd hdf5/
git checkout tags/hdf5-1_10_2
./configure --enable-parallel --enable-shared --prefix=/usr/local CC=/usr/local/bin/mpicc CXX=/usr/local/bin/mpic++
make -j
sudo make -j install

cd ~/install
git clone https://github.com/NanoComp/harminv.git
cd harminv/
sh autogen.sh --enable-shared
make -j
sudo make -j install

cd ~/install
git clone https://github.com/NanoComp/libctl.git
cd libctl/
sh autogen.sh  --enable-shared
make -j
sudo make -j install

cd ~/install
git clone https://github.com/NanoComp/h5utils.git
cd h5utils/
sh autogen.sh CC=/usr/local/bin/mpicc LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}"
make -j
sudo make -j install

cd ~/install
git clone https://github.com/NanoComp/mpb.git
cd mpb/
sh autogen.sh --enable-shared CC=/usr/local/bin/mpicc LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}" --with-hermitian-eps
make -j
sudo make -j install

cd ~/install
git clone https://github.com/HomerReid/libGDSII.git
cd libGDSII/
sh autogen.sh
sudo make -j install

cd ~/install
wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.0.tar.gz
tar xvf mpi4py-3.0.0.tar.gz
cd mpi4py-3.0.0/
python3 setup.py build
sudo /usr/local/bin/python3 setup.py install

cd ~/install
wget https://github.com/h5py/h5py/archive/2.8.0.tar.gz
tar xvf 2.8.0.tar.gz
cd h5py-2.8.0/
python3 setup.py configure --mpi
python3 setup.py build
sudo /usr/local/bin/python3 setup.py install

cd ~/install
git clone https://github.com/NanoComp/meep.git
cd meep/
sh autogen.sh --enable-shared --with-mpi --with-openmp PYTHON=python3 MPICC=/usr/local/bin/mpicc MPICXX=/usr/local/bin/mpic++ LDFLAGS="${MY_LDFLAGS}" CPPFLAGS="${MY_CPPFLAGS}"
make -j
sudo make install
```

Meep for Developers
-------------------

If you want to modify the source code, you will want to have a number of additional packages, most importantly the [Git](https://git-scm.com/) version-control system.

Once you have Git, you can obtain a copy of the source repository from GitHub:

```sh
 git clone https://github.com/NanoComp/meep.git
```

This command clones the repository in a local directory `meep`. See the [Git manual](https://git-scm.com/doc) for more information on using Git. Perhaps the most useful command is `git pull` which you can execute periodically to get any new updates to the development version.

Git will give you an absolutely minimal set of sources; to create a usable directory, you should run:

```sh
sh autogen.sh
make
```

in the `meep` directory. And subsequently, if you are editing the sources you should include `--enable-maintainer-mode` whenever you reconfigure. To do this, however, you will need a number of additional packages beyond those listed above:

-   GNU [autoconf](https://www.gnu.org/software/autoconf/autoconf.html), [automake](https://www.gnu.org/software/automake/), and [libtool](https://www.gnu.org/software/libtool/libtool.html) &mdash; these are used to create the Makefiles and configure scripts, and to build shared libraries.
-   [SWIG](http://www.swig.org/) &mdash; the Python and Scheme interfaces are largely generated by a program called *SWIG* (Simple Wrapper and Interface Generator).
