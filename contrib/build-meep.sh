#!/bin/bash

# Latest version of this script can be found at:
#   https://github.com/NanoComp/meep/blob/master/contrib/build-meep.sh

# included LD_PRELOAD for CentOS is needed otherwise
# this error message is displayed when starting python:
#   python3 /tmp/test-meep.py
#   ** failed to load python MPI module (mpi4py)
#   ** /usr/local/lib64/python3.6/site-packages/mpi4py/MPI.cpython-36m-x86_64-linux-gnu.so: undefined symbol: ompi_mpi_logical8

help ()
{
    cat << EOF

$1: Download MEEP sources and dependencies, build, and install

Usage: $1 [options]
EOF
    sed -ne 's,^[ \t]*\(-[^ \t]*\))[^#]*#[ \t]*\(.*\),    \1 \2,p' "$1"
    echo ""
    echo "After installation, environment file 'meep-env.sh' is created in destination path."
    echo ""
    exit 1
}

gitclone ()
{
    repo=${1##*/}
    name=${repo%%.*}
    echo $repo $name
    if [ -d $name ]; then
        ( cd $name; git pull; )
    else
        [ -z "$2" ] || branch="-b $2"
        git clone --depth=1 $1 $branch
    fi
}

autogensh ()
{
    LIB64="${DESTDIR}/lib"
    $centos && LIB64="${DESTDIR}/lib64"
    LLP="${LD_LIBRARY_PATH}:${LIB64}"
    set -x
    sh autogen.sh PKG_CONFIG_PATH="${PKG_CONFIG_PATH}" RPATH_FLAGS="${RPATH_FLAGS}" \
        PYTHON=python3 CC="${CC}" LDFLAGS="${LDFLAGS}" CFLAGS="${CFLAGS}" CPPFLAGS="${CPPFLAGS}" LD_LIBRARY_PATH=${LLP} \
        --enable-shared --prefix="${DESTDIR}" --libdir=${LIB64} \
        --with-libctl=${DESTDIR}/share/libctl \
        "$@"
    set -x
}

showenv()
{
    echo export PATH=${DESTDIR}/bin:\${PATH}
    echo export LD_LIBRARY_PATH=${DESTDIR}/lib:\${LD_LIBRARY_PATH}
    echo export PYTHONPATH=${DESTDIR}/lib/${python}/site-packages:\${PYTHONPATH}
    if $centos; then
        echo export LD_LIBRARY_PATH=${DESTDIR}/lib64:\${LD_LIBRARY_PATH}
        echo export PYTHONPATH=/usr/local/lib64/${python}/site-packages:\${PYTHONPATH}
        echo export LD_PRELOAD+=:/usr/lib64/openmpi/lib/libmpi.so
    fi
}

buildinstall=true
installdeps=true
bashrc=false
BLAS=""
unset DESTDIR

while [ ! -z "$1" ]; do
    case "$1" in
        -h)         # help
            help "$0"
            ;;
        -d)         # <installdir>  (mandatory)
            DESTDIR="$2"
            shift
            ;;
        -s)         # use 'sudo' for 'make install'
            SUDO=sudo
            ;;
        -S)         # source directory (default: <installdir>/src)
            SRCDIR="$2"
            shift
            ;;
        -n)         # skip checking for distribution dependencies
            installdeps=false
            ;;
        -c)         # skip build+install
            buildinstall=false
            ;;
        -Du1604)    # build 'meep-ubuntu:16.04' docker image
            docker=ubuntu:16.04;;
        -Du1804)    # build 'meep-ubuntu:18.04' docker image
            docker=ubuntu:18.04;;
        -Dcentos7)  # build 'meep-centos:7' docker image
            docker=centos:7;;
        -Batlas)    # blas: use atlas
            BLAS=atlas;;
        -Bopenblas) # blas: use openblas
            BLAS=openblas;;
        -Bgslcblas)      # blas: use gsl+openblas (test)
            BLAS=gslcblas;;
        --bashrc)
            bashrc=true;; # undocumented internal to store env in ~/.bashrc
        *)
            echo "'$1' ?"
            help "$0"
            ;;
    esac
    shift
done

$buildinstall && [ -z "${DESTDIR}" ] && { echo "-d option is missing" ; help "$0"; }
$buildinstall && [ -z "${BLAS}" ] && { echo "blas taste is missing" ; help "$0"; }

if [ ! -z "${docker}" ]; then
    ddir="docker-${docker}"
    mkdir ${ddir}
    cp "$0" ${ddir}/
    cd ${ddir}
    case ${docker} in
        *ubuntu*)
            echo "FROM ${docker}" > Dockerfile
            echo "RUN apt-get update && apt-get -y install apt-utils sudo" >> Dockerfile
            echo "ADD \"$0\" \"$0\"" >> Dockerfile
            echo "RUN mkdir -p ${DESTDIR}; ./\"$0\" -d ${DESTDIR} --bashrc -B${BLAS}" >> Dockerfile
            echo "CMD /bin/bash" >> Dockerfile
            exec docker build -t "meep-${docker}-${BLAS}" .
            ;;

        *centos*)
            echo "FROM ${docker}" > Dockerfile
            echo "RUN yum -y install sudo" >> Dockerfile
            echo "ADD \"$0\" \"$0\"" >> Dockerfile
            echo "RUN mkdir -p ${DESTDIR}; ./\"$0\" -d ${DESTDIR} --bashrc -B${BLAS}" >> Dockerfile
            echo "CMD /bin/bash" >> Dockerfile
            exec docker build -t "meep-${docker}-${BLAS}" .
            exit 1;;

        *)
            echo "can't build a docker file for '${docker}'"
            help "$0"
            exit 1;;
    esac
fi

# detect wether DESTDIR is ending with src/
[ "${DESTDIR##*/}" = src ] && DESTDIR=$(cd $(pwd)/..; pwd)
[ -z "$SRCDIR" ] && SRCDIR=${DESTDIR}/src

if [ ! -r /etc/os-release ]; then
    echo "Error: cannot read /etc/os-release"
    false
fi

set -e

ubuntu=false
centos=false

. /etc/os-release
distrib="${ID}${VERSION_ID}"
case "$distrib" in
    ubuntu18.04) # ubuntu 18.04 bionic
        libpng=libpng-dev
        libpython=libpython3-dev
        python=python3.6
        ubuntu=true
        ;;
    ubuntu16.04) # ubuntu 16.04 xenial
        libpng=libpng16-dev
        libpython=libpython3.5-dev
        python=python3.5
        ubuntu=true
        ;;
    centos7) # CentOS 7.x
        python=python3.6
        centos=true
        ;;
    *)
        echo "Error: unsupported distribution '${distrib}'"
        false
        ;;
esac

# these are passed to configure on demand with: 'autogensh ... CC=${CC} CXX=${CXX}'
export CC=mpicc
export CXX=mpicxx
export CFLAGS="-O3 -mtune=native"

if $ubuntu; then
    RPATH_FLAGS="-Wl,-rpath,${DESTDIR}/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi"
    LDFLAGS="-L${DESTDIR}/lib -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi ${RPATH_FLAGS}"
    CFLAGS="${CFLAGS} -I${DESTDIR}/include -I/usr/include/hdf5/openmpi"
fi

if $centos; then
    # mpicc is not in PATH
    export CC=/usr/lib64/openmpi/bin/mpicc
    export CXX=/usr/lib64/openmpi/bin/mpicxx

    RPATH_FLAGS="-Wl,-rpath,${DESTDIR}/lib64:/usr/lib64/openmpi/lib"
    LDFLAGS="-L${DESTDIR}/lib64 -L/usr/lib64/openmpi/lib ${RPATH_FLAGS}"
    CFLAGS="${CFLAGS} -I${DESTDIR}/include -I/usr/include/openmpi-x86_64/"
fi

eval $(showenv)

if $installdeps && $ubuntu; then

    sudo apt-get update

    case ${BLAS} in
        atlas*) BLAS=atlas; bpkg=libatlas-base-dev;;
        openblas) bpkg=libopenblas-dev;;
        gslcblas) BLAS=openblas; bpkg="libgsl-dev libgslcblas0 libopenblas-dev";;
    esac

    sudo apt-get -y install     \
        git                     \
        build-essential         \
        gfortran                \
        libgmp-dev              \
        swig                    \
        autoconf                \
        pkg-config              \
        $libpng                 \
        git                     \
        guile-2.0-dev           \
        libfftw3-dev            \
        libhdf5-openmpi-dev     \
        hdf5-tools              \
        $libpython              \
        python3-numpy           \
        python3-scipy           \
        python3-pip             \
        ffmpeg                  \
        ${bpkg}                 \

fi

if $installdeps && $centos; then

    sudo yum -y --enablerepo=extras install epel-release

    case ${BLAS} in
        atlas) bpkg="atlas atlas-devel"; LDFLAGS="${LDFLAGS} -L/usr/lib64/${BLAS}"; BLAS=tatlas;;
        gslcblas|openblas) bpkg=openblas-devel; BLAS=openblas;;
    esac

    sudo yum -y install    \
        git                \
        bison              \
        byacc              \
        cscope             \
        ctags              \
        cvs                \
        diffstat           \
        oxygen             \
        flex               \
        gcc                \
        gcc-c++            \
        gcc-gfortran       \
        gettext            \
        git                \
        indent             \
        intltool           \
        libtool            \
        patch              \
        patchutils         \
        redhat-rpm-config  \
        rpm-build          \
        systemtap          \
        wget               \
        python3            \
        python3-devel      \
        python36-numpy     \
        python36-scipy     \
        fftw3-devel        \
        libpng-devel       \
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
        ffmpeg             \
        openmpi-devel      \
        hdf5-openmpi-devel \
        guile-devel        \
        swig               \
        ${bpkg}

fi

CPPFLAGS=${CFLAGS}
PKG_CONFIG_PATH=${DESDTIR}/pkgconfig
export PKG_CONFIG_PATH
export PATH=${DESTDIR}/bin:${PATH}

if $buildinstall; then

    mkdir -p ${SRCDIR}

    cd ${SRCDIR}
    gitclone https://github.com/NanoComp/harminv.git
    cd harminv/
    autogensh --with-blas=${BLAS}
    make -j && $SUDO make install

    cd ${SRCDIR}
    gitclone https://github.com/NanoComp/libctl.git
    cd libctl/
    autogensh --with-blas=${BLAS}
    make -j && $SUDO make install

    cd ${SRCDIR}
    gitclone https://github.com/NanoComp/h5utils.git
    cd h5utils/
    autogensh CC=${CC}
    make -j && $SUDO make install

    cd ${SRCDIR}
    gitclone https://github.com/NanoComp/mpb.git
    cd mpb/
    autogensh CC=${CC} --with-hermitian-eps --with-blas=${BLAS}
    make -j && $SUDO make install

    cd ${SRCDIR}
    gitclone https://github.com/HomerReid/libGDSII.git
    cd libGDSII/
    autogensh
    make -j && $SUDO make install

if $ubuntu; then
    sudo -E -H python3 -m pip install --upgrade pip
    sudo -E -H python3 -m pip install --no-cache-dir mpi4py
    export HDF5_MPI="ON" # for python h5py
    sudo -E -H python3 -m pip install cython
    sudo -E -H python3 -m pip install --no-cache-dir --no-binary=h5py h5py
    sudo -E -H python3 -m pip install --no-cache-dir matplotlib>3.0.0
else
    sudo -E -H python3 -m pip install --no-cache-dir mpi4py
    #sudo -E -H python3 -m pip install --no-binary=h5py h5py
    sudo -E -H python3 -m pip install h5py
    sudo -E -H python3 -m pip install matplotlib>3.0.0
fi

    cd ${SRCDIR}
    gitclone https://github.com/NanoComp/meep.git
    cd meep/
    autogensh --with-mpi --with-openmp --with-blas=${BLAS}
    make -j && $SUDO make install

    # all done

    if $centos; then
         cd ${DESTDIR}/lib/${python}/site-packages/meep/
         for i in ../../../../lib64/${python}/site-packages/meep/*meep*; do
            ln -sf $i
         done
    fi

fi # buildinstall

########
# test

test=/tmp/test-meep.py

cat << EOF > $test
import meep as mp
cell = mp.Vector3(16,8,0)
print(cell)
exit()
EOF

echo "------------ ENV (commands)"
showenv > ${DESTDIR}/meep-env.sh
$bashrc && { showenv >> ~/.bashrc; }
echo "------------ ENV (result)"
echo export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
echo export PYTHONPATH=${PYTHONPATH}
echo export LD_PRELOAD=${LD_PRELOAD}
echo "------------ $test"
cat $test
echo "------------ EXEC python3 $test"
. ${DESTDIR}/meep-env.sh
python3 $test
