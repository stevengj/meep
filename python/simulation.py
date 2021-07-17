# -*- coding: utf-8 -*-
from __future__ import division, print_function

import functools
import math
import numbers
import os
import re
import signal
import subprocess
import sys
import warnings
from collections import namedtuple
from collections import OrderedDict

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy as np

import meep as mp
from meep.geom import Vector3, init_do_averaging
from meep.source import EigenModeSource, GaussianBeamSource, IndexedSource, check_positive
import meep.visualization as vis
from meep.verbosity_mgr import Verbosity

try:
    basestring
except NameError:
    basestring = str

try:
    from ipywidgets import FloatProgress
    from IPython.display import display
    do_progress = True
except ImportError:
    do_progress = False

verbosity = Verbosity(mp.cvar, 'meep', 1)


# Send output from Meep, ctlgeom, and MPB to Python's stdout
mp.set_meep_printf_callback(mp.py_master_printf_wrap)
mp.set_meep_printf_stderr_callback(mp.py_master_printf_stderr_wrap)
mp.set_ctl_printf_callback(mp.py_master_printf_wrap)
mp.set_mpb_printf_callback(mp.py_master_printf_wrap)

EigCoeffsResult = namedtuple('EigCoeffsResult', ['alpha', 'vgrp', 'kpoints', 'kdom', 'cscale'])
FluxData = namedtuple('FluxData', ['E', 'H'])
ForceData = namedtuple('ForceData', ['offdiag1', 'offdiag2', 'diag'])
NearToFarData = namedtuple('NearToFarData', ['F'])

def fix_dft_args(args, i):
    if len(args) > i+2 and isinstance(args[i],(int,float)) and isinstance(args[i+1],(int,float)) and isinstance(args[i+2],int):
        fcen = args[i]
        df = args[i+1]
        nfreq = args[i+2]
        freq = [fcen] if nfreq == 1 else np.linspace(fcen-0.5*df,fcen+0.5*df,nfreq)
        return args[:i] + (freq,) + args[i+3:]
    elif not isinstance(args[i], (np.ndarray, list)):
        raise TypeError("add_dft functions only accept fcen,df,nfreq (3 numbers) or freq (array/list)")
    else:
        return args

def get_num_args(func):
    if isinstance(func, Harminv):
        return 2
    return func.__code__.co_argcount


def vec(*args):
    try:
        # Check for vec(x, [y, [z]])
        return mp._vec(*args)
    except (TypeError, NotImplementedError):
        try:
            # Check for vec(iterable)
            if len(args) != 1:
                raise TypeError

            return mp._vec(*args[0])
        except (TypeError, NotImplementedError):
            print("Expected an iterable with three or fewer floating point values")
            print("    or something of the form vec(x, [y, [z]])")
            raise


def py_v3_to_vec(dims, iterable, is_cylindrical=False):
    v3 = Vector3(*iterable)
    if dims == 1:
        return mp.vec(v3.z)
    elif dims == 2:
        if is_cylindrical:
            return mp.veccyl(v3.x, v3.z)
        else:
            v = mp.vec(v3.x, v3.y)
            v.set_direction(mp.Z, v3.z) # for special_kz handling
            return v
    elif dims == 3:
        return mp.vec(v3.x, v3.y, v3.z)
    else:
        raise ValueError("Invalid dimensions in Volume: {}".format(dims))

def bands_to_diffractedplanewave(where,bands):
    if bands.axis is None:
        if where.in_direction(mp.X) != 0:
            axis = np.array([1, 0, 0], dtype=np.float64)
        elif where.in_direction(mp.Y) != 0:
            axis = np.array([0, 1, 0], dtype=np.float64)
        elif where.in_direction(mp.Z) != 0:
            axis = np.array([0, 0, 1], dtype=np.float64)
        else:
            raise ValueError("axis parameter of DiffractedPlanewave must be a non-zero Vector3")
    elif isinstance(bands.axis,mp.Vector3):
        axis = np.array([bands.axis.x, bands.axis.y, bands.axis.z], dtype=np.float64)
    else:
        raise TypeError("axis parameter of DiffractedPlanewave must be a Vector3")
    diffractedplanewave_args = [
        np.array(bands.g, dtype=np.intc),
        axis,
        bands.s * 1.0,
        bands.p * 1.0
    ]
    return mp.diffractedplanewave(*diffractedplanewave_args)

class DiffractedPlanewave(object):
    """
    For mode decomposition or eigenmode source, specify a diffracted planewave in homogeneous media. Should be passed as the `bands` argument of `get_eigenmode_coefficients`, `band_num` of `get_eigenmode`, or `eig_band` of `EigenModeSource`.
    """
    def __init__(self,
                 g=None,
                 axis=None,
                 s=None,
                 p=None):
        """
        Construct a `DiffractedPlanewave`.

        + **`g` [ list of 3 `integer`s ]** — The diffraction order $(m_x,m_y,m_z)$ corresponding to the wavevector $(k_x+2\\pi m_x/\\Lambda_x,k_y+2\\pi m_y/\\Lambda_y,k_z+2\\pi m_z/\\Lambda_z)$. The diffraction order $m_{x,y,z}$ should be non-zero only in the $d$-1 periodic directions of a $d$ dimensional cell (e.g., a plane in 3d) in which the mode monitor or source extends the entire length of the cell.

        + **`axis` [ `Vector3` ]** — The plane of incidence for each planewave (used to define the $\\mathcal{S}$ and $\\mathcal{P}$ polarizations below) is defined to be the plane that contains the `axis` vector and the planewave's wavevector. If `None`, `axis` defaults to the first direction that lies in the plane of the monitor or source (e.g., $y$ direction for a $yz$ plane in 3d, either $x$ or $y$ in 2d).

        + **`s` [ `complex` ]** — The complex amplitude of the $\\mathcal{S}$ polarziation (i.e., electric field perpendicular to the plane of incidence).

        + **`p` [ `complex` ]** — The complex amplitude of the $\\mathcal{P}$ polarziation (i.e., electric field parallel to the plane of incidence).
        """
        self._g = g
        self._axis = axis
        self._s = complex(s)
        self._p = complex(p)

    @property
    def g(self):
        return self._g

    @property
    def axis(self):
        return self._axis

    @property
    def s(self):
        return self._s

    @property
    def p(self):
        return self._p

DefaultPMLProfile = lambda u: u * u

class PML(object):
    """
    This class is used for specifying the PML absorbing boundary layers around the cell,
    if any, via the `boundary_layers` input variable. See also [Perfectly Matched
    Layers](Perfectly_Matched_Layer.md). `boundary_layers` can be zero or more `PML`
    objects, with multiple objects allowing you to specify different PML layers on
    different boundaries. The class represents a single PML layer specification, which
    sets up one or more PML layers around the boundaries according to the following
    properties.
    """
    def __init__(self, thickness,
                 direction=mp.ALL,
                 side=mp.ALL,
                 R_asymptotic=1e-15,
                 mean_stretch=1.0,
                 pml_profile=DefaultPMLProfile):
        """
        + **`thickness` [`number`]** — The spatial thickness of the PML layer which
          extends from the boundary towards the *inside* of the cell. The thinner it is,
          the more numerical reflections become a problem. No default value.

        + **`direction` [`direction` constant ]** — Specify the direction of the
          boundaries to put the PML layers next to. e.g. if `X`, then specifies PML on the
          $\\pm x$ boundaries (depending on the value of `side`, below). Default is the
          special value `ALL`, which puts PML layers on the boundaries in all directions.

        + **`side` [`side` constant ]** — Specify which side, `Low` or `High` of the
          boundary or boundaries to put PML on. e.g. if side is `Low` and direction is
          `X`, then a PML layer is added to the $-x$ boundary. Default is the special
          value `ALL`, which puts PML layers on both sides.

        + **`R_asymptotic` [`number`]** — The asymptotic reflection in the limit of
          infinite resolution or infinite PML thickness, for reflections from air (an
          upper bound for other media with index &gt; 1). For a finite resolution or
          thickness, the reflection will be *much larger*, due to the discretization of
          Maxwell's equation. Default value is 10<sup>−15</sup>, which should suffice for
          most purposes. You want to set this to be small enough so that waves propagating
          within the PML are attenuated sufficiently, but making `R_asymptotic` too small
          will increase the numerical reflection due to discretization.

        + **`pml_profile` [`function`]** — By default, Meep turns on the PML conductivity
          quadratically within the PML layer &mdash; one doesn't want to turn it on
          suddenly, because that exacerbates reflections due to the discretization. More
          generally, with `pml_profile` one can specify an arbitrary PML "profile"
          function $f(u)$ that determines the shape of the PML absorption profile up to an
          overall constant factor. *u* goes from 0 to 1 at the start and end of the PML,
          and the default is $f(u) = u^2$. In some cases where a very thick PML is
          required, such as in a periodic medium (where there is technically no such thing
          as a true PML, only a pseudo-PML), it can be advantageous to turn on the PML
          absorption more smoothly. See [Optics Express, Vol. 16, pp. 11376-92,
          2008](http://www.opticsinfobase.org/abstract.cfm?URI=oe-16-15-11376). For
          example, one can use a cubic profile $f(u) = u^3$ by specifying
          `pml_profile=lambda u: u*u*u`.
        """
        self.thickness = thickness
        self.direction = direction
        self.side = side
        self.R_asymptotic = R_asymptotic
        self.mean_stretch = mean_stretch
        self.pml_profile = pml_profile

        if direction == mp.ALL and side == mp.ALL:
            self.swigobj = mp.pml(thickness, R_asymptotic, mean_stretch)
        elif direction == mp.ALL:
            self.swigobj = mp.pml(thickness, side, R_asymptotic, mean_stretch)
        else:
            self.swigobj = mp.pml(thickness, direction, side, R_asymptotic, mean_stretch)

    @property
    def R_asymptotic(self):
        return self._R_asymptotic

    @R_asymptotic.setter
    def R_asymptotic(self, val):
        self._R_asymptotic = check_positive('PML.R_asymptotic', val)

    @property
    def mean_stretch(self):
        return self._mean_stretch

    @mean_stretch.setter
    def mean_stretch(self, val):
        if val >= 1:
            self._mean_stretch = val
        else:
            raise ValueError("PML.mean_stretch must be >= 1. Got {}".format(val))


class Absorber(PML):
    """
    Instead of a `PML` layer, there is an alternative class called `Absorber` which is a
    **drop-in** replacement for `PML`. For example, you can do
    `boundary_layers=[mp.Absorber(thickness=2)]` instead of
    `boundary_layers=[mp.PML(thickness=2)]`. All the parameters are the same as for `PML`,
    above. You can have a mix of `PML` on some boundaries and `Absorber` on others.

    The `Absorber` class does *not* implement a perfectly matched layer (PML), however
    (except in 1d). Instead, it is simply a scalar electric **and** magnetic conductivity
    that turns on gradually within the layer according to the `pml_profile` (defaulting to
    quadratic). Such a scalar conductivity gradient is only reflectionless in the limit as
    the layer becomes sufficiently thick.

    The main reason to use `Absorber` is if you have **a case in which PML fails:**

    -   No true PML exists for *periodic* media, and a scalar absorber is computationally
        less expensive and generally just as good. See [Optics Express, Vol. 16, pp.
        11376-92, 2008](http://www.opticsinfobase.org/abstract.cfm?URI=oe-16-15-11376).
    -   PML can lead to *divergent* fields for certain waveguides with "backward-wave"
        modes; this can readily occur in metals with surface plasmons, and a scalar
        absorber is your only choice. See [Physical Review E, Vol. 79, 065601,
        2009](http://math.mit.edu/~stevenj/papers/LohOs09.pdf).
    -   PML can fail if you have a waveguide hitting the edge of your cell *at an angle*.
        See [J. Computational Physics, Vol. 230, pp. 2369-77,
        2011](http://math.mit.edu/~stevenj/papers/OskooiJo11.pdf).
    """



class Symmetry(object):
    """
    This class is used for the `symmetries` input variable to specify symmetries which
    must preserve both the structure *and* the sources. Any number of symmetries can be
    exploited simultaneously but there is no point in specifying redundant symmetries: the
    cell can be reduced by at most a factor of 4 in 2d and 8 in 3d. See also [Exploiting
    Symmetry](Exploiting_Symmetry.md). This is the base class of the specific symmetries
    below, so normally you don't create it directly. However, it has two properties which
    are shared by all symmetries:

    The specific symmetry sub-classes are:

    **`Mirror`** — A mirror symmetry plane. `direction` is the direction *normal* to the
    mirror plane.

    **`Rotate2`** — A 180° (twofold) rotational symmetry (a.k.a. $C_2$). `direction` is
    the axis of the rotation.

    **`Rotate4`** — A 90° (fourfold) rotational symmetry (a.k.a. $C_4$). `direction` is
    the axis of the rotation.
    """
    def __init__(self, direction, phase=1):
        """
        Construct a `Symmetry`.

        + **`direction` [`direction` constant ]** — The direction of the symmetry (the
          normal to a mirror plane or the axis for a rotational symmetry). e.g. `X`, `Y`,
          or `Z` (only Cartesian/grid directions are allowed). No default value.

        + **`phase` [`complex`]** — An additional phase to multiply the fields by when
          operating the symmetry on them. Default is +1, e.g. a phase of -1 for a mirror
          plane corresponds to an *odd* mirror. Technically, you are essentially
          specifying the representation of the symmetry group that your fields and sources
          transform under.
        """
        self.direction = direction
        self.phase = complex(phase)
        self.swigobj = None


class Rotate2(Symmetry):
    """
    A 180° (twofold) rotational symmetry (a.k.a. $C_2$). `direction` is the axis of the
    rotation.
    """


class Rotate4(Symmetry):
    """
    A 90° (fourfold) rotational symmetry (a.k.a. $C_4$). `direction` is the axis of the
    rotation.
    """


class Mirror(Symmetry):
    """
    A mirror symmetry plane. `direction` is the direction *normal* to the mirror plane.
    """


class Identity(Symmetry):
    """
    """


class Volume(object):
    """
    Many Meep functions require you to specify a volume in space, corresponding to the C++
    type `meep::volume`. This class creates such a volume object, given the `center` and
    `size` properties (just like e.g. a `Block` object). If the `size` is not specified,
    it defaults to `(0,0,0)`, i.e. a single point. Any method that accepts such a volume
    also accepts `center` and `size` keyword arguments. If these are specified instead of
    the volume, the library will construct a volume for you. Alternatively, you can
    specify a list of `Vector3` vertices using the `vertices` parameter. The `center` and
    `size` will automatically be computed from this list.
    """

    def __init__(self, center=Vector3(), size=Vector3(), dims=2, is_cylindrical=False, vertices=[]):
        """
        Construct a Volume.
        """
        if len(vertices) == 0:
            self.center = Vector3(*center)
            self.size = Vector3(*size)
        else:
            vertices = np.array([np.array(i) for i in vertices])
            self.center = Vector3(*np.mean(vertices,axis=0))
            x_list = np.unique(vertices[:,0])
            y_list = np.unique(vertices[:,1])
            z_list = np.unique(vertices[:,2])

            x_size = 0 if x_list.size == 1 else np.abs(np.diff(x_list)[0])
            y_size = 0 if y_list.size == 1 else np.abs(np.diff(y_list)[0])
            z_size = 0 if z_list.size == 1 else np.abs(np.diff(z_list)[0])

            self.size = Vector3(x_size,y_size,z_size)

        self.dims = dims

        v1 = self.center - self.size.scale(0.5)
        v2 = self.center + self.size.scale(0.5)

        vec1 = py_v3_to_vec(self.dims, v1, is_cylindrical)
        vec2 = py_v3_to_vec(self.dims, v2, is_cylindrical)

        self.swigobj = mp.volume(vec1, vec2)

    def get_vertices(self):
        xmin = self.center.x - self.size.x/2
        xmax = self.center.x + self.size.x/2
        ymin = self.center.y - self.size.y/2
        ymax = self.center.y + self.size.y/2
        zmin = self.center.z - self.size.z/2
        zmax = self.center.z + self.size.z/2

        # Iterate over and remove duplicates for collapsed dimensions (i.e. min=max))
        return [Vector3(x,y,z) for x in list(set([xmin,xmax])) for y in list(set([ymin,ymax])) for z in list(set([zmin,zmax]))]


    def get_edges(self):
        vertices = self.get_vertices()
        edges = []

        # Useful for importing weird geometries and the sizes are slightly off
        def nearly_equal(a,b,sig_fig=10):
            return a==b or (abs(a-b) < 10**(-sig_fig))

        for iter1 in range(len(vertices)):
            for iter2 in range(iter1+1,len(vertices)):
                if ((iter1 != iter2) and
                nearly_equal((vertices[iter1]-vertices[iter2]).norm(),self.size.x) or
                nearly_equal((vertices[iter1]-vertices[iter2]).norm(),self.size.y) or
                nearly_equal((vertices[iter1]-vertices[iter2]).norm(),self.size.z)):
                    edges.append([vertices[iter1],vertices[iter2]])
        return edges

    def pt_in_volume(self,pt):
        xmin = self.center.x - self.size.x/2
        xmax = self.center.x + self.size.x/2
        ymin = self.center.y - self.size.y/2
        ymax = self.center.y + self.size.y/2
        zmin = self.center.z - self.size.z/2
        zmax = self.center.z + self.size.z/2

        if (pt.x >= xmin and pt.x <= xmax and pt.y >= ymin and pt.y <= ymax and pt.z >= zmin and pt.z <= zmax):
            return True
        else:
            return False


class FluxRegion(object):
    """
    A `FluxRegion` object is used with [`add_flux`](#flux-spectra) to specify a region in
    which Meep should accumulate the appropriate Fourier-transformed fields in order to
    compute a flux spectrum. It represents a region (volume, plane, line, or point) in
    which to compute the integral of the Poynting vector of the Fourier-transformed
    fields. `ModeRegion` is an alias for `FluxRegion` for use with `add_mode_monitor`.

    Note that the flux is always computed in the *positive* coordinate direction, although
    this can effectively be flipped by using a `weight` of -1.0. This is useful, for
    example, if you want to compute the outward flux through a box, so that the sides of
    the box add instead of subtract.
    """
    def __init__(self, center=None, size=Vector3(), direction=mp.AUTOMATIC, weight=1.0, volume=None):
        """
        Construct a `FluxRegion` object.

        + **`center` [`Vector3`]** — The center of the flux region (no default).

        + **`size` [`Vector3`]** — The size of the flux region along each of the coordinate
          axes. Default is `(0,0,0)`; a single point.

        + **`direction` [`direction` constant ]** — The direction in which to compute the
          flux (e.g. `mp.X`, `mp.Y`, etcetera). Default is `AUTOMATIC`, in which the
          direction is determined by taking the normal direction if the flux region is a
          plane (or a line, in 2d). If the normal direction is ambiguous (e.g. for a point
          or volume), then you *must* specify the `direction` explicitly (not doing so
          will lead to an error).

        + **`weight` [`complex`]** — A weight factor to multiply the flux by when it is
          computed. Default is 1.0.

        + **`volume` [`Volume`]** — A `meep.Volume` can be used to specify the flux region
          instead of a `center` and a `size`.
        """
        if center is None and volume is None:
            raise ValueError("Either center or volume required")

        if volume:
            self.center = volume.center
            self.size = volume.size
        else:
            self.center = Vector3(*center)
            self.size = Vector3(*size)

        self.direction = direction
        self.weight = complex(weight)


ModeRegion = FluxRegion
Near2FarRegion = FluxRegion


class ForceRegion(FluxRegion):
    """
    A region (volume, plane, line, or point) in which to compute the integral of the
    stress tensor of the Fourier-transformed fields. Its properties are:

    + **`center` [ `Vector3` ]** — The center of the force region (no default).

    + **`size` [ `Vector3` ]** — The size of the force region along each of the coordinate
      axes. Default is `(0,0,0)` (a single point).

    + **`direction` [ `direction constant` ]** — The direction of the force that you wish
      to compute (e.g. `X`, `Y`, etcetera). Unlike `FluxRegion`, you must specify this
      explicitly, because there is not generally any relationship between the direction of
      the force and the orientation of the force region.

    + **`weight` [ `complex` ]** — A weight factor to multiply the force by when it is
      computed. Default is 1.0.

    + **`volume` [`Volume`]** — A `meep.Volume` can be used to specify the force region
      instead of a `center` and a `size`.

    In most circumstances, you should define a set of `ForceRegion`s whose union is a
    closed surface lying in vacuum and enclosing the object that is experiencing the
    force.
    """


class EnergyRegion(FluxRegion):
    """
    A region (volume, plane, line, or point) in which to compute the integral of the
    energy density of the Fourier-transformed fields. Its properties are:

    + **`center` [`Vector3`]** — The center of the energy region (no default).

    + **`size` [`Vector3`]** — The size of the energy region along each of the coordinate
      axes. Default is (0,0,0): a single point.

    + **`weight` [`complex`]** — A weight factor to multiply the energy density by when it
      is computed. Default is 1.0.
    """


class FieldsRegion(object):

    def __init__(self, where=None, center=None, size=None):
        if where:
            self.center = where.center
            self.size = where.size
        else:
            self.center = Vector3(*center) if center is not None else None
            self.size = Vector3(*size) if size is not None else None

        self.where = where


class DftObj(object):
    """Wrapper around dft objects that allows delayed initialization of the structure.

    When splitting the structure into chunks for parallel simulations, we want to know all
    of the details of the simulation in order to ensure that each processor gets a similar
    amount of work. The problem with DFTs is that the 'add_flux' style methods immediately
    initialize the structure and fields. So, if the user adds multiple DFT objects to the
    simulation, the load balancing code only knows about the first one and can't split the
    work up nicely. To circumvent this, we delay the execution of the 'add_flux' methods
    as late as possible. When 'add_flux' (or add_near2far, etc.) is called, we:

    1. Create an instance of the appropriate subclass of DftObj (DftForce, DftFlux, etc.).
       Set its args property to the list of arguments passed to add_flux, and set its func
       property to the 'real' add_flux, which is prefixed by an underscore.

    2. Add this DftObj to the list Simulation.dft_objects. When we actually run the
       simulation, we call Simulation._evaluate_dft_objects, which calls dft.func(*args)
       for each dft in the list.

    If the user tries to access a property or call a function on the DftObj before
    Simulation._evaluate_dft_objects is called, then we initialize the C++ object through
    swigobj_attr and return the property they requested.
    """
    def __init__(self, func, args):
        """Construct a `DftObj`."""
        self.func = func
        self.args = args
        self.swigobj = None

    def swigobj_attr(self, attr):
        if self.swigobj is None:
            self.swigobj = self.func(*self.args)
        return getattr(self.swigobj, attr)

    @property
    def save_hdf5(self):
        return self.swigobj_attr('save_hdf5')

    @property
    def load_hdf5(self):
        return self.swigobj_attr('load_hdf5')

    @property
    def scale_dfts(self):
        return self.swigobj_attr('scale_dfts')

    @property
    def remove(self):
        return self.swigobj_attr('remove')

    @property
    def freq(self):
        return self.swigobj_attr('freq')

    @property
    def where(self):
        return self.swigobj_attr('where')


class DftFlux(DftObj):
    """
    """

    def __init__(self, func, args):
        """Construct a `DftFlux`."""
        super(DftFlux, self).__init__(func, args)
        self.nfreqs = len(args[0])
        self.regions = args[1]
        self.num_components = 4

    @property
    def flux(self):
        return self.swigobj_attr('flux')

    @property
    def E(self):
        return self.swigobj_attr('E')

    @property
    def H(self):
        return self.swigobj_attr('H')

    @property
    def cE(self):
        return self.swigobj_attr('cE')

    @property
    def cH(self):
        return self.swigobj_attr('cH')

    @property
    def normal_direction(self):
        return self.swigobj_attr('normal_direction')

    @property
    def freq(self):
        return self.swigobj_attr('freq')

class DftForce(DftObj):
    """
    """

    def __init__(self, func, args):
        """Construct a `DftForce`."""
        super(DftForce, self).__init__(func, args)
        self.nfreqs = len(args[0])
        self.regions = args[1]
        self.num_components = 6

    @property
    def force(self):
        return self.swigobj_attr('force')

    @property
    def offdiag1(self):
        return self.swigobj_attr('offdiag1')

    @property
    def offdiag2(self):
        return self.swigobj_attr('offdiag2')

    @property
    def diag(self):
        return self.swigobj_attr('diag')

    @property
    def freq(self):
        return self.swigobj_attr('freq')

class DftNear2Far(DftObj):
    """
    """

    def __init__(self, func, args):
        """Construct a `DftNear2Far`."""
        super(DftNear2Far, self).__init__(func, args)
        self.nfreqs = len(args[0])
        self.nperiods = args[1]
        self.regions = args[2]
        self.num_components = 4

    @property
    def farfield(self):
        return self.swigobj_attr('farfield')

    @property
    def save_farfields(self):
        return self.swigobj_attr('save_farfields')

    @property
    def F(self):
        return self.swigobj_attr('F')

    @property
    def eps(self):
        return self.swigobj_attr('eps')

    @property
    def mu(self):
        return self.swigobj_attr('mu')

    def flux(self, direction, where, resolution):
        """
        Given a `Volume` `where` (may be 0d, 1d, 2d, or 3d) and a `resolution` (in grid
        points / distance unit), compute the far fields in `where` (which may lie
        *outside* the cell) in a grid with the given resolution (which may differ from the
        FDTD solution) and return its Poynting flux in `direction` as a list. The dataset
        is a 1d array of `nfreq` dimensions.
        """
        return self.swigobj_attr('flux')(direction, where.swigobj, resolution)

    @property
    def freq(self):
        return self.swigobj_attr('freq')

class DftEnergy(DftObj):
    """
    """

    def __init__(self, func, args):
        """Construct a `DftEnergy`."""
        super(DftEnergy, self).__init__(func, args)
        self.nfreqs = len(args[0])
        self.regions = args[1]
        self.num_components = 12

    @property
    def electric(self):
        return self.swigobj_attr('electric')

    @property
    def magnetic(self):
        return self.swigobj_attr('magnetic')

    @property
    def total(self):
        return self.swigobj_attr('total')

    @property
    def freq(self):
        return self.swigobj_attr('freq')

class DftFields(DftObj):
    """
    """

    def __init__(self, func, args):
        """Construct a `DftFields`."""
        super(DftFields, self).__init__(func, args)
        self.nfreqs = len(args[4])
        self.regions = [FieldsRegion(where=args[1], center=args[2], size=args[3])]
        self.num_components = len(args[0])

    @property
    def chunks(self):
        return self.swigobj_attr('chunks')


Mode = namedtuple('Mode', ['freq', 'decay', 'Q', 'amp', 'err'])


class EigenmodeData(object):

    def __init__(self, band_num, freq, group_velocity, k, swigobj, kdom):
        """Construct an `EigenmodeData`."""
        self.band_num = band_num
        self.freq = freq
        self.group_velocity = group_velocity
        self.k = k
        self.swigobj = swigobj
        self.kdom = kdom

    def amplitude(self, point, component):
        swig_point = mp.vec(point.x, point.y, point.z)
        return mp.eigenmode_amplitude(self.swigobj, swig_point, component)


class Harminv(object):
    """
    `Harminv` is implemented as a class with a [`__call__`](#Harminv.__call__) method,
    which allows it to be used as a step function that collects field data from a given
    point and runs [Harminv](https://github.com/NanoComp/harminv) on that data to extract
    the frequencies, decay rates, and other information.

    See [`__init__`](#Harminv.__init__) for details about constructing a `Harminv`.

    **Important:** normally, you should only use Harminv to analyze data *after the
    sources are off*. Wrapping it in `after_sources(mp.Harminv(...))` is sufficient.

    In particular, Harminv takes the time series $f(t)$ corresponding to the given field
    component as a function of time and decomposes it (within the specified bandwidth) as:

    $$f(t) = \\sum_n a_n e^{-i\\omega_n t}$$

    The results are stored in the list `Harminv.modes`, which is a list of tuples holding
    the frequency, amplitude, and error of the modes. Given one of these tuples (e.g.,
    `first_mode = harminv_instance.modes[0]`), you can extract its various components:

    + **`freq`** — The real part of frequency ω (in the usual Meep 2πc units).

    + **`decay`** — The imaginary part of the frequency ω.

    + **`Q`** — The dimensionless lifetime, or quality factor defined as
      $-\\mathrm{Re}\\,\\omega / 2 \\mathrm{Im}\\,\\omega$.

    + **`amp`** — The complex amplitude $a$.

    + **`err`** — A crude measure of the error in the frequency (both real and imaginary).
      If the error is much larger than the imaginary part, for example, then you can't
      trust the $Q$ to be accurate. Note: this error is only the uncertainty in the signal
      processing, and tells you nothing about the errors from finite resolution, finite
      cell size, and so on.

    For example, `[m.freq for m in harminv_instance.modes]` gives a list of the real parts
    of the frequencies. Be sure to save a reference to the `Harminv` instance if you wish
    to use the results after the simulation:

    ```py
    sim = mp.Simulation(...)
    h = mp.Harminv(...)
    sim.run(mp.after_sources(h))
    # do something with h.modes
    ```
    """
    def __init__(self, c, pt, fcen, df, mxbands=None):
        """
        Construct a Harminv object.

        A `Harminv` is a step function that collects data from the field component `c`
        (e.g. $E_x$, etc.) at the given point `pt` (a `Vector3`). Then, at the end
        of the run, it uses Harminv to look for modes in the given frequency range (center
        `fcen` and width `df`), printing the results to standard output (prefixed by
        `harminv:`) as comma-delimited text, and also storing them to the variable
        `Harminv.modes`. The optional argument `mxbands` is the maximum number of modes to
        search for. Defaults to 100.
        """
        self.c = c
        self.pt = pt
        self.fcen = fcen
        self.df = df
        self.mxbands = mxbands
        self.data = []
        self.data_dt = 0
        self.modes = []
        self.spectral_density = 1.1
        self.Q_thresh = 50.0
        self.rel_err_thresh = mp.inf
        self.err_thresh = 0.01
        self.rel_amp_thresh = -1.0
        self.amp_thresh = -1.0
        self.step_func = self._harminv()

    def __call__(self, sim, todo):
        """
        Allows a Haminv instance to be used as a step function.
        """
        self.step_func(sim, todo)

    def _collect_harminv(self):

        def _collect1(c, pt):
            self.t0 = 0

            def _collect2(sim):
                self.data_dt = sim.meep_time() - self.t0
                self.t0 = sim.meep_time()
                self.data.append(sim.get_field_point(c, pt))
            return _collect2
        return _collect1

    def _check_freqs(self, sim):
        source_freqs = [(s.src.frequency, 0 if s.src.width == 0 else 1 / s.src.width)
                        for s in sim.sources
                        if hasattr(s.src, 'frequency')]

        harminv_max = self.fcen + 0.5 * self.df
        harminv_min = self.fcen - 0.5 * self.df

        for sf in source_freqs:
            sf_max = sf[0] + 0.5 * sf[1]
            sf_min = sf[0] - 0.5 * sf[1]
            if harminv_max > sf_max:
                warn_fmt = "Harminv frequency {} is outside maximum Source frequency {}"
                warnings.warn(warn_fmt.format(harminv_max, sf_max), RuntimeWarning)
            if harminv_min < sf_min:
                warn_fmt = "Harminv frequency {} is outside minimum Source frequency {}"
                warnings.warn(warn_fmt.format(harminv_min, sf_min), RuntimeWarning)

    def _analyze_harminv(self, sim, maxbands):
        harminv_cols = ['frequency', 'imag. freq.', 'Q', '|amp|', 'amplitude', 'error']
        display_run_data(sim, 'harminv', harminv_cols)
        self._check_freqs(sim)

        dt = self.data_dt if self.data_dt is not None else sim.fields.dt

        bands = mp.py_do_harminv(self.data, dt, self.fcen - self.df / 2, self.fcen + self.df / 2, maxbands,
                                 self.spectral_density, self.Q_thresh, self.rel_err_thresh, self.err_thresh,
                                 self.rel_amp_thresh, self.amp_thresh)

        modes = []
        for freq, amp, err in bands:
            Q = freq.real / (-2 * freq.imag) if freq.imag != 0 else float('inf')
            modes.append(Mode(freq.real, freq.imag, Q, amp, err))
            display_run_data(sim, 'harminv', [freq.real, freq.imag, Q, abs(amp), amp, err])

        return modes

    def _harminv(self):

        def _harm(sim):

            if self.mxbands is None or self.mxbands == 0:
                mb = 100
            else:
                mb = self.mxbands
            self.modes = self._analyze_harminv(sim, mb)

        f1 = self._collect_harminv()

        return _combine_step_funcs(at_end(_harm), f1(self.c, self.pt))


class Simulation(object):
    """
    The `Simulation` [class](#classes) contains all the attributes that you can set to
    control various parameters of the Meep computation.
    """
    def __init__(self,
                 cell_size,
                 resolution,
                 geometry=[],
                 sources=[],
                 eps_averaging=True,
                 dimensions=3,
                 boundary_layers=[],
                 symmetries=[],
                 force_complex_fields=False,
                 default_material=mp.Medium(),
                 m=0,
                 k_point=False,
                 kz_2d="complex",
                 extra_materials=[],
                 material_function=None,
                 epsilon_func=None,
                 epsilon_input_file='',
                 progress_interval=4,
                 subpixel_tol=1e-4,
                 subpixel_maxeval=100000,
                 ensure_periodicity=True,
                 num_chunks=0,
                 Courant=0.5,
                 accurate_fields_near_cylorigin=False,
                 filename_prefix=None,
                 output_volume=None,
                 output_single_precision=False,
                 load_structure='',
                 geometry_center=mp.Vector3(),
                 force_all_components=False,
                 split_chunks_evenly=True,
                 chunk_layout=None,
                 collect_stats=False):
        """
        All `Simulation` attributes are described in further detail below. In brackets
        after each variable is the type of value that it should hold. The classes, complex
        datatypes like `GeometricObject`, are described in a later subsection. The basic
        datatypes, like `integer`, `boolean`, `complex`, and `string` are defined by
        Python. `Vector3` is a `meep` class.

        + **`geometry` [ list of `GeometricObject` class ]** — Specifies the geometric
          objects making up the structure being simulated. When objects overlap, later
          objects in the list take precedence. Defaults to no objects (empty list).

        + **`geometry_center` [ `Vector3` class ]** — Specifies the coordinates of the
          center of the cell. Defaults to (0, 0, 0), but changing this allows you to shift
          the coordinate system used in Meep (for example, to put the origin at the
          corner).  Passing `geometry_center=c` is equivalent to adding the `c` vector to
          the coordinates of every other object in the simulation, i.e. `c` becomes the
          new origin that other objects are defined with respect to.

        + **`sources` [ list of `Source` class ]** — Specifies the current sources to be
          present in the simulation. Defaults to none (empty list).

        + **`symmetries` [ list of `Symmetry` class ]** — Specifies the spatial symmetries
          (mirror or rotation) to exploit in the simulation. Defaults to none (empty
          list). The symmetries must be obeyed by *both* the structure and the sources.
          See also [Exploiting Symmetry](Exploiting_Symmetry.md).

        + **`boundary_layers` [ list of `PML` class ]** — Specifies the
          [PML](Perfectly_Matched_Layer.md) absorbing boundary layers to use. Defaults to
          none.

        + **`cell_size` [ `Vector3` ]** — Specifies the size of the cell which is centered
          on the origin of the coordinate system. Any sizes of 0 imply a
          reduced-dimensionality calculation. Strictly speaking, the dielectric function
          is taken to be uniform along that dimension. A 2d calculation is especially
          optimized. See `dimensions` below. **Note:** because Maxwell's equations are
          scale invariant, you can use any units of distance you want to specify the cell
          size: nanometers, microns, centimeters, etc. However, it is usually convenient
          to pick some characteristic lengthscale of your problem and set that length to 1.
          See also [Units](Introduction.md#units-in-meep). Required argument (no default).

        + **`default_material` [`Medium` class ]** — Holds the default material that is
          used for points not in any object of the geometry list. Defaults to `air` (ε=1).
          This can also be a NumPy array that defines a dielectric function much like
          `epsilon_input_file` below (see below). If you want to use a material function
          as the default material, use the `material_function` keyword argument (below).

        + **`material_function` [ function ]** — A Python function that takes a `Vector3`
          and returns a `Medium`. See also [Material Function](#medium).
          Defaults to `None`.

        + **`epsilon_func` [ function ]** — A Python function that takes a `Vector3` and
          returns the dielectric constant at that point. See also [Material
          Function](#medium). Defaults to `None`.

        + **`epsilon_input_file` [`string`]** — If this string is not empty (the default),
          then it should be the name of an HDF5 file whose first/only dataset defines a
          scalar, real-valued, frequency-independent dielectric function over some
          discrete grid. Alternatively, the dataset name can be specified explicitly if
          the string is in the form "filename:dataset". This dielectric function is then
          used in place of the ε property of `default_material` (i.e. where there are no
          `geometry` objects). The grid of the epsilon file dataset need *not* match the
          computational grid; it is scaled and/or linearly interpolated as needed to map
          the file onto the cell. The structure is warped if the proportions of the grids
          do not match. **Note:** the file contents only override the ε property of the
          `default_material`, whereas other properties (μ, susceptibilities,
          nonlinearities, etc.) of `default_material` are still used.

        + **`dimensions` [`integer`]** — Explicitly specifies the dimensionality of the
          simulation, if the value is less than 3. If the value is 3 (the default), then
          the dimensions are automatically reduced to 2 if possible when `cell_size` in
          the $z$ direction is `0`. If `dimensions` is the special value of `CYLINDRICAL`,
          then cylindrical coordinates are used and the $x$ and $z$ dimensions are
          interpreted as $r$ and $z$, respectively. If `dimensions` is 1, then the cell
          must be along the $z$ direction and only $E_x$ and $H_y$ field components are
          permitted. If `dimensions` is 2, then the cell must be in the $xy$ plane.

        + **`m` [`number`]** — For `CYLINDRICAL` simulations, specifies that the angular
          $\\phi$ dependence of the fields is of the form $e^{im\\phi}$ (default is `m=0`).
          If the simulation cell includes the origin $r=0$, then `m` must be an integer.

        + **`accurate_fields_near_cylorigin` [`boolean`]** — For `CYLINDRICAL` simulations
          with |*m*| &gt; 1, compute more accurate fields near the origin $r=0$ at the
          expense of requiring a smaller Courant factor. Empirically, when this option is
          set to `True`, a Courant factor of roughly $\\min[0.5, 1 / (|m| + 0.5)]$ or
          smaller seems to be needed. Default is `False`, in which case the $D_r$, $D_z$,
          and $B_r$ fields within |*m*| pixels of the origin are forced to zero, which
          usually ensures stability with the default Courant factor of 0.5, at the expense
          of slowing convergence of the fields near $r=0$.

        + **`resolution` [`number`]** — Specifies the computational grid resolution in
          pixels per distance unit. Required argument. No default.

        + **`k_point` [`False` or `Vector3`]** — If `False` (the default), then the
          boundaries are perfect metallic (zero electric field). If a `Vector3`, then the
          boundaries are Bloch-periodic: the fields at one side are
          $\\exp(i\\mathbf{k}\\cdot\\mathbf{R})$ times the fields at the other side, separated
          by the lattice vector $\\mathbf{R}$. A non-zero `Vector3` will produce complex
          fields. The `k_point` vector is specified in Cartesian coordinates in units of
          2π/distance. Note: this is *different* from [MPB](https://mpb.readthedocs.io),
          equivalent to taking MPB's `k_points` through its function
          `reciprocal->cartesian`.

        + **`kz_2d` [`"complex"`, `"real/imag"`, or `"3d"`]** — A 2d cell (i.e.,
          `dimensions=2`) combined with a `k_point` that has a *non-zero* component in $z$
          would normally result in a 3d simulation with complex fields. However, by
          default (`kz_2d="complex"`), Meep will use a 2d computational cell in which
          $k_z$ is incorporated as an additional term in Maxwell's equations, which still
          results in complex fields but greatly improved performance. Setting `kz_2d="3d"`
          will instead use a 3d cell that is one pixel thick (with Bloch-periodic boundary
          conditions), which is considerably more expensive. The third possibility,
          `kz_2d="real/imag"`, saves an additional factor of two by storing some field
          components as purely real and some as purely imaginary in a "real" field, but
          this option requires some care to use. See [2d Cell with Out-of-Plane
          Wavevector](2d_Cell_Special_kz.md).

        + **`ensure_periodicity` [`boolean`]** — If `True` (the default) *and* if the
          boundary conditions are periodic (`k_point` is not `False`), then the geometric
          objects are automatically repeated periodically according to the lattice vectors
          which define the size of the cell.

        + **`eps_averaging` [`boolean`]** — If `True` (the default), then [subpixel
          averaging](Subpixel_Smoothing.md) is used when initializing the dielectric
          function. For simulations involving a [material function](#medium),
          `eps_averaging` is `False` (the default) and must be
          [enabled](Subpixel_Smoothing.md#enabling-averaging-for-material-function) in
          which case the input variables `subpixel_maxeval` (default 10<sup>4</sup>) and
          `subpixel_tol` (default 10<sup>-4</sup>) specify the maximum number of function
          evaluations and the integration tolerance for the adaptive numerical
          integration. Increasing/decreasing these, respectively, will cause a more
          accurate but slower computation of the average ε with diminishing returns for
          the actual FDTD error. Disabling subpixel averaging will lead to [staircasing
          effects and irregular
          convergence](Subpixel_Smoothing.md#what-happens-when-subpixel-smoothing-is-disabled).

        + **`force_complex_fields` [`boolean`]** — By default, Meep runs its simulations
          with purely real fields whenever possible. It uses complex fields which require
          twice the memory and computation if the `k_point` is non-zero or if `m` is
          non-zero. However, by setting `force_complex_fields` to `True`, Meep will always
          use complex fields.

        + **`force_all_components` [`boolean`]** — By default, in a 2d simulation Meep
          uses only the field components that might excited by your current sources:
          either the in-plane $(E_x,E_y,H_z)$ or out-of-plane $(H_x,H_y,E_z)$ polarization,
          depending on the source.  (Both polarizations are excited if you use multiple source
          polarizations, or if an anisotropic medium is present that couples the two
          polarizations.)   In rare cases (primarily for combining results of multiple
          simulations with differing polarizations), you might want to force it to
          simulate all fields, even those that remain zero throughout the simulation, by
          setting `force_all_components` to `True`.

        + **`filename_prefix` [`string`]** — A string prepended to all output filenames
          (e.g., for HDF5 files). If `None` (the default), then Meep constructs a default
          prefix based on the current Python filename ".py" replaced by "-" (e.g. `foo.py`
          uses a `"foo-"` prefix). You can get this prefix by calling `get_filename_prefix`.

        + **`Courant` [`number`]** — Specify the
          [Courant factor](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition)
          $S$ which relates the time step size to the spatial discretization: $cΔ t = SΔ x$.
          Default is 0.5. For numerical stability, the Courant factor must be *at
          most* $n_\\textrm{min}/\\sqrt{\\textrm{# dimensions}}$, where $n_\\textrm{min}$ is
          the minimum refractive index (usually 1), and in practice $S$ should be slightly
          smaller.

        + **`output_volume` [`Volume` class ]** — Specifies the default region of space
          that is output by the HDF5 output functions (below); see also the `Volume` class
          which manages `meep::volume*` objects. Default is `None`, which means that the
          whole cell is output. Normally, you should use the `in_volume(...)` function to
          modify the output volume instead of setting `output_volume` directly.

        + **`output_single_precision` [`boolean`]** — Meep performs its computations in
          [double precision](https://en.wikipedia.org/wiki/double_precision), and by
          default its output HDF5 files are in the same format. However, by setting this
          variable to `True` (default is `False`) you can instead output in [single
          precision](https://en.wikipedia.org/wiki/single_precision) which saves a factor
          of two in space.

        + **`progress_interval` [`number`]** — Time interval (seconds) after which Meep
          prints a progress message. Default is 4 seconds.

        + **`extra_materials` [ list of `Medium` class ]** — By default, Meep turns off
          support for material dispersion ([susceptibilities](#susceptibility) or
          [conductivity](Materials.md#conductivity-and-complex)) or nonlinearities if none
          of the objects in `geometry` have materials with these properties &mdash; since
          they are not needed, it is faster to omit their calculation. This doesn't work,
          however, if you use a `material_function`: materials via a user-specified
          function of position instead of just geometric objects. If your material
          function only returns a nonlinear material, for example, Meep won't notice this
          unless you tell it explicitly via `extra_materials`. `extra_materials` is a list
          of materials that Meep should look for in the cell in addition to any materials
          that are specified by geometric objects. You should list any materials other
          than scalar dielectrics that are returned by `material_function` here.

        + **`load_structure` [`string`]** — If not empty, Meep will load the structure
          file specified by this string. The file must have been created by
          `mp.dump_structure`. Defaults to an empty string. See [Load and Dump
          Structure](#load-and-dump-structure) for more information.

        + **`chunk_layout` [`string` or `Simulation` instance or `BinaryPartition` class]** —
          This will cause the `Simulation` to use the chunk layout described by either
          (1) an `.h5` file (created using `Simulation.dump_chunk_layout`), (2) another
          `Simulation` instance, or (3) a [`BinaryPartition`](#binarypartition) class object.
          For more information, see [Load and Dump Structure](#load-and-dump-structure) and
          [Parallel Meep/User-Specified Cell Partition](Parallel_Meep.md#user-specified-cell-partition).

        The following require a bit more understanding of the inner workings of Meep to
        use. See also [SWIG Wrappers](#swig-wrappers).

        + **`structure` [`meep::structure*`]** — Pointer to the current structure being
          simulated; initialized by `_init_structure` which is called automatically by
          `init_sim()` which is called automatically by any of the [run
          functions](#run-functions). The structure initialization is handled by the
          `Simulation` class, and most users will not need to call `_init_structure`.

        + **`fields` [`meep::fields*`]** — Pointer to the current fields being simulated;
          initialized by `init_sim()` which is called automatically by any of the [run
          functions](#run-functions).

        + **`num_chunks` [`integer`]** — Minimum number of "chunks" (subregions) to divide
          the structure/fields into. Overrides the default value determined by
          the number of processors, PML layers, etcetera. Mainly useful for debugging.

        + **`split_chunks_evenly` [`boolean`]** — When `True` (the default), the work per
          [chunk](Chunks_and_Symmetry.md) is not taken into account when splitting chunks
          up for multiple processors. The cell is simply split up into equal chunks (with
          the exception of PML regions, which must be on their own chunk). When `False`,
          Meep attempts to allocate an equal amount of work to each processor, which can
          increase the performance of [parallel simulations](Parallel_Meep.md).
        """

        self.cell_size = Vector3(*cell_size)
        self.geometry = geometry
        self.sources = sources
        self.resolution = resolution
        self.dimensions = dimensions
        self.boundary_layers = boundary_layers
        self.symmetries = symmetries
        self.geometry_center = Vector3(*geometry_center)
        self.eps_averaging = eps_averaging
        self.subpixel_tol = subpixel_tol
        self.subpixel_maxeval = subpixel_maxeval
        self.ensure_periodicity = ensure_periodicity
        self.extra_materials = extra_materials
        self.default_material = default_material
        self.epsilon_input_file = epsilon_input_file
        self.num_chunks = chunk_layout.numchunks() if isinstance(chunk_layout,mp.BinaryPartition) else num_chunks
        self.Courant = Courant
        self.global_d_conductivity = 0
        self.global_b_conductivity = 0
        self.k_point = k_point
        self.fields = None
        self.structure = None
        self.accurate_fields_near_cylorigin = accurate_fields_near_cylorigin
        self.m = m
        self.force_complex_fields = force_complex_fields
        self.progress_interval = progress_interval
        self.init_sim_hooks = []
        self.run_index = 0
        self.filename_prefix = filename_prefix
        self.output_append_h5 = None
        self.output_single_precision = output_single_precision
        self.output_volume = output_volume
        self.last_eps_filename = ''
        self.output_h5_hook = lambda fname: False
        self.interactive = False
        self.is_cylindrical = False
        self.material_function = material_function
        self.epsilon_func = epsilon_func
        self.load_structure_file = load_structure
        self.dft_objects = []
        self._is_initialized = False
        self.force_all_components = force_all_components
        self.split_chunks_evenly = split_chunks_evenly
        self.chunk_layout = chunk_layout
        self.collect_stats = collect_stats
        self.fragment_stats = None
        self._output_stats = os.environ.get('MEEP_STATS', None)

        self.special_kz = False
        if self.cell_size.z == 0 and self.k_point and self.k_point.z != 0:
            if kz_2d == "complex":
                self.special_kz = True
                self.force_complex_fields = True
            elif kz_2d == "real/imag":
                self.special_kz = True
                self.force_complex_fields = False
            elif kz_2d == "3d":
                self.special_kz = False
            else:
                raise ValueError("Invalid kz_2d option: {} not in [complex, real/imag, 3d]".format(kz_2d))

    # To prevent the user from having to specify `dims` and `is_cylindrical`
    # to Volumes they create, the library will adjust them appropriately based
    # on the settings in the Simulation instance. This method must be called on
    # any user-defined Volume before passing it to meep via its `swigobj`.
    def _fit_volume_to_simulation(self, vol):
        return Volume(vol.center, vol.size, dims=self.dimensions, is_cylindrical=self.is_cylindrical)

    # Every function that takes a user volume can be specified either by a volume
    # (a Python Volume or a SWIG-wrapped meep::volume), or a center and a size
    def _volume_from_kwargs(self, vol=None, center=None, size=None):
        if vol:
            if isinstance(vol, Volume):
                # A pure Python Volume
                return self._fit_volume_to_simulation(vol).swigobj
            else:
                # A SWIG-wrapped meep::volume
                return vol
        elif size is not None and center is not None:
            return Volume(center=Vector3(*center), size=Vector3(*size), dims=self.dimensions,
                          is_cylindrical=self.is_cylindrical).swigobj
        else:
            raise ValueError("Need either a Volume, or a size and center")

    def _infer_dimensions(self, k):
        if self.dimensions == 3:

            def use_2d(self, k):
                zero_z = self.cell_size.z == 0
                return zero_z and (not k or self.special_kz or k.z == 0)

            if use_2d(self, k):
                return 2
            else:
                return 3
        return self.dimensions

    def _get_valid_material_frequencies(self):
        fmin = float('-inf')
        fmax = float('inf')

        all_materials = [go.material for go in self.geometry] + self.extra_materials
        all_materials.append(self.default_material)

        for mat in all_materials:
            if isinstance(mat, mp.Medium) and mat.valid_freq_range:
                if mat.valid_freq_range.min > fmin:
                    fmin = mat.valid_freq_range.min
                if mat.valid_freq_range.max < fmax:
                    fmax = mat.valid_freq_range.max

        return fmin, fmax

    def _check_material_frequencies(self):

        min_freq, max_freq = self._get_valid_material_frequencies()
        source_freqs = [(s.src.frequency, 0 if s.src.width == 0 else 1 / s.src.width)
                        for s in self.sources
                        if hasattr(s.src, 'frequency')]

        dft_freqs = []
        for dftf in self.dft_objects:
            dft_freqs.append(dftf.freq[0])
            dft_freqs.append(dftf.freq[-1])

        warn_src = ('Note: your sources include frequencies outside the range of validity of the ' +
                    'material models. This is fine as long as you eventually only look at outputs ' +
                    '(fluxes, resonant modes, etc.) at valid frequencies.')

        warn_dft_fmt = "DFT frequency {} is out of material's range of {}-{}"

        for sf in source_freqs:
            if sf[0] + 0.5 * sf[1] > max_freq or sf[0] - 0.5 * sf[1] < min_freq:
                warnings.warn(warn_src, RuntimeWarning)

        for dftf in dft_freqs:
            if dftf > max_freq or dftf < min_freq:
                warnings.warn(warn_dft_fmt.format(dftf, min_freq, max_freq), RuntimeWarning)

    def _create_grid_volume(self, k):
        dims = self._infer_dimensions(k)

        if dims == 0 or dims == 1:
            gv = mp.vol1d(self.cell_size.z, self.resolution)
        elif dims == 2:
            self.dimensions = 2
            gv = mp.vol2d(self.cell_size.x, self.cell_size.y, self.resolution)
        elif dims == 3:
            gv = mp.vol3d(self.cell_size.x, self.cell_size.y, self.cell_size.z, self.resolution)
        elif dims == mp.CYLINDRICAL:
            gv = mp.volcyl(self.cell_size.x, self.cell_size.z, self.resolution)
            self.dimensions = 2
            self.is_cylindrical = True
        else:
            raise ValueError("Unsupported dimentionality: {}".format(dims))

        gv.center_origin()
        gv.shift_origin(py_v3_to_vec(self.dimensions, self.geometry_center, self.is_cylindrical))
        return gv

    def _create_symmetries(self, gv):
        sym = mp.symmetry()

        # Initialize swig objects for each symmetry and combine them into one
        for s in self.symmetries:
            if isinstance(s, Identity):
                s.swigobj = mp.identity()
            elif isinstance(s, Rotate2):
                s.swigobj = mp.rotate2(s.direction, gv)
                sym += s.swigobj * complex(s.phase.real, s.phase.imag)
            elif isinstance(s, Rotate4):
                s.swigobj = mp.rotate4(s.direction, gv)
                sym += s.swigobj * complex(s.phase.real, s.phase.imag)
            elif isinstance(s, Mirror):
                s.swigobj = mp.mirror(s.direction, gv)
                sym += s.swigobj * complex(s.phase.real, s.phase.imag)
            else:
                s.swigobj = mp.symmetry()

        return sym

    def _get_dft_volumes(self):
        volumes = [self._volume_from_kwargs(vol=r.where if hasattr(r, 'where') else None,
                                            center=r.center, size=r.size)
                   for dft in self.dft_objects
                   for r in dft.regions]

        return volumes

    def _boundaries_to_vols_1d(self, boundaries):
        v1 = []

        for bl in boundaries:
            cen = mp.Vector3(z=(self.cell_size.z / 2) - (0.5 * bl.thickness))
            sz = mp.Vector3(z=bl.thickness)
            if bl.side == mp.High or bl.side == mp.ALL:
                v1.append(self._volume_from_kwargs(center=cen, size=sz))
            if bl.side == mp.Low or bl.side == mp.ALL:
                v1.append(self._volume_from_kwargs(center=-1 * cen, size=sz))

        return v1

    def _boundaries_to_vols_2d_3d(self, boundaries, cyl=False):
        side_thickness = OrderedDict()
        side_thickness['top'] = 0
        side_thickness['bottom'] = 0
        side_thickness['left'] = 0
        side_thickness['right'] = 0
        side_thickness['near'] = 0
        side_thickness['far'] = 0

        for bl in boundaries:
            d = bl.direction
            s = bl.side
            if d == mp.X or d == mp.ALL:
                if s == mp.High or s == mp.ALL:
                    side_thickness['right'] = bl.thickness
                if s == mp.Low or s == mp.ALL:
                    side_thickness['left'] = bl.thickness
            if d == mp.Y or d == mp.ALL:
                if s == mp.High or s == mp.ALL:
                    side_thickness['top'] = bl.thickness
                if s == mp.Low or s == mp.ALL:
                    side_thickness['bottom'] = bl.thickness
            if self.dimensions == 3:
                if d == mp.Z or d == mp.ALL:
                    if s == mp.High or s == mp.ALL:
                        side_thickness['far'] = bl.thickness
                    if s == mp.Low or s == mp.ALL:
                        side_thickness['near'] = bl.thickness

        xmax = self.cell_size.x / 2
        ymax = self.cell_size.z / 2 if cyl else self.cell_size.y / 2
        zmax = self.cell_size.z / 2
        ytot = self.cell_size.z if cyl else self.cell_size.y

        def get_overlap_0(side, d):
            if side == 'top' or side == 'bottom':
                ydir = 1 if side == 'top' else -1
                xsz = self.cell_size.x - (side_thickness['left'] + side_thickness['right'])
                ysz = d
                zsz = self.cell_size.z - (side_thickness['near'] + side_thickness['far'])
                xcen = xmax - side_thickness['right'] - (xsz / 2)
                ycen = ydir*ymax + (-ydir*0.5*d)
                zcen = zmax - side_thickness['far'] - (zsz / 2)
            elif side == 'left' or side == 'right':
                xdir = 1 if side == 'right' else -1
                xsz = d
                ysz = ytot - (side_thickness['top'] + side_thickness['bottom'])
                zsz = self.cell_size.z - (side_thickness['near'] + side_thickness['far'])
                xcen = xdir*xmax + (-xdir*0.5*d)
                ycen = ymax - side_thickness['top'] - (ysz / 2)
                zcen = zmax - side_thickness['far'] - (zsz / 2)
            elif side == 'near' or side == 'far':
                zdir = 1 if side == 'far' else -1
                xsz = self.cell_size.x - (side_thickness['left'] + side_thickness['right'])
                ysz = ytot - (side_thickness['top'] + side_thickness['bottom'])
                zsz = d
                xcen = xmax - side_thickness['right'] - (xsz / 2)
                ycen = ymax - side_thickness['top'] - (ysz / 2)
                zcen = zdir*zmax + (-zdir*0.5*d)

            if cyl:
                cen = mp.Vector3(xcen, 0, ycen)
                sz = mp.Vector3(xsz, 0, ysz)
            else:
                cen = mp.Vector3(xcen, ycen, zcen)
                sz = mp.Vector3(xsz, ysz, zsz)

            return self._volume_from_kwargs(center=cen, size=sz)

        def get_overlap_1(side1, side2, d):
            if side_thickness[side2] == 0:
                return []

            if side1 == 'top' or side1 == 'bottom':
                ydir = 1 if side1 == 'top' else -1
                ysz = d
                ycen = ydir*ymax + (-ydir*0.5*d)
                if side2 == 'left' or side2 == 'right':
                    xdir = 1 if side2 == 'right' else -1
                    xsz = side_thickness[side2]
                    zsz = self.cell_size.z - (side_thickness['near'] + side_thickness['far'])
                    xcen = xdir*xmax + (-xdir*0.5*side_thickness[side2])
                    zcen = zmax - side_thickness['far'] - (zsz / 2)
                elif side2 == 'near' or side2 == 'far':
                    zdir = 1 if side2 == 'far' else -1
                    xsz = self.cell_size.x - (side_thickness['left'] + side_thickness['right'])
                    zsz = side_thickness[side2]
                    xcen = xmax - side_thickness['right'] - (xsz / 2)
                    zcen = zdir*zmax + (-zdir*0.5*side_thickness[side2])
            elif side1 == 'near' or side1 == 'far':
                xdir = 1 if side2 == 'right' else -1
                zdir = 1 if side1 == 'far' else -1
                xsz = side_thickness[side2]
                ysz = self.cell_size.y - (side_thickness['top'] + side_thickness['bottom'])
                zsz = d
                xcen = xdir*xmax + (-xdir*0.5*side_thickness[side2])
                ycen = ymax - side_thickness['top'] - (ysz / 2)
                zcen = zdir*zmax + (-zdir*0.5*d)

            if cyl:
                cen = mp.Vector3(xcen, 0, ycen)
                sz = mp.Vector3(xsz, 0, ysz)
            else:
                cen = mp.Vector3(xcen, ycen, zcen)
                sz = mp.Vector3(xsz, ysz, zsz)
            return self._volume_from_kwargs(center=cen, size=sz)

        def get_overlap_2(side1, side2, side3, d):
            if side_thickness[side2] == 0 or side_thickness[side3] == 0:
                return []
            xdir = 1 if side2 == 'right' else -1
            ydir = 1 if side1 == 'top' else -1
            zdir = 1 if side3 == 'far' else -1
            xsz = side_thickness[side2]
            ysz = d
            zsz = side_thickness[side3]
            xcen = xdir*xmax + (-xdir*0.5*xsz)
            ycen = ydir*ymax + (-ydir*0.5*d)
            zcen = zdir*zmax + (-zdir*0.5*zsz)

            cen = mp.Vector3(xcen, ycen, zcen)
            sz = mp.Vector3(xsz, ysz, zsz)
            return self._volume_from_kwargs(center=cen, size=sz)

        v1 = []
        v2 = []
        v3 = []

        for side, thickness in side_thickness.items():
            if thickness == 0:
                continue

            v1.append(get_overlap_0(side, thickness))
            if side == 'top' or side == 'bottom':
                v2.append(get_overlap_1(side, 'left', thickness))
                v2.append(get_overlap_1(side, 'right', thickness))
                if self.dimensions == 3:
                    v2.append(get_overlap_1(side, 'near', thickness))
                    v2.append(get_overlap_1(side, 'far', thickness))
                    v3.append(get_overlap_2(side, 'left', 'near', thickness))
                    v3.append(get_overlap_2(side, 'right', 'near', thickness))
                    v3.append(get_overlap_2(side, 'left', 'far', thickness))
                    v3.append(get_overlap_2(side, 'right', 'far', thickness))
            if side == 'near' or side == 'far':
                v2.append(get_overlap_1(side, 'left', thickness))
                v2.append(get_overlap_1(side, 'right', thickness))

        return [v for v in v1 if v], [v for v in v2 if v], [v for v in v3 if v]

    def _boundary_layers_to_vol_list(self, boundaries):
        """
        Returns three lists of meep::volume objects. The first represents the boundary
        regions with no overlaps. The second is regions where two boundaries overlap, and
        the third is regions where three boundaries overlap
        """

        vols1 = []
        vols2 = []
        vols3 = []

        if self.dimensions == 1:
            vols1 = self._boundaries_to_vols_1d(boundaries)
        else:
            vols1, vols2, vols3 = self._boundaries_to_vols_2d_3d(boundaries, self.is_cylindrical)

        return vols1, vols2, vols3

    def _make_fragment_lists(self, gv):

        def convert_volumes(dft_obj):
            volumes = []
            for r in dft_obj.regions:
                volumes.append(self._volume_from_kwargs(vol=r.where if hasattr(r, 'where') else None,
                                                        center=r.center, size=r.size))
            return volumes

        dft_data_list = [mp.dft_data(o.nfreqs, o.num_components, convert_volumes(o))
                         for o in self.dft_objects]

        pmls = []
        absorbers = []
        for bl in self.boundary_layers:
            if type(bl) is PML:
                pmls.append(bl)
            elif type(bl) is Absorber:
                absorbers.append(bl)

        pml_vols1, pml_vols2, pml_vols3 = self._boundary_layers_to_vol_list(pmls)
        absorber_vols1, absorber_vols2, absorber_vols3 = self._boundary_layers_to_vol_list(absorbers)
        absorber_vols = absorber_vols1 + absorber_vols2 + absorber_vols3

        return (dft_data_list, pml_vols1, pml_vols2, pml_vols3, absorber_vols)

    def _compute_fragment_stats(self, gv):

        dft_data_list, pml_vols1, pml_vols2, pml_vols3, absorber_vols = self._make_fragment_lists(gv)

        stats = mp.compute_fragment_stats(
            self.geometry,
            gv,
            self.cell_size,
            self.geometry_center,
            self.default_material,
            dft_data_list,
            pml_vols1,
            pml_vols2,
            pml_vols3,
            absorber_vols,
            self.extra_materials,
            self.subpixel_tol,
            self.subpixel_maxeval,
            self.ensure_periodicity,
            self.eps_averaging
        )

        mirror_symmetries = [sym for sym in self.symmetries if isinstance(sym, Mirror)]
        for sym in mirror_symmetries:
            stats.num_anisotropic_eps_pixels //= 2
            stats.num_anisotropic_mu_pixels //= 2
            stats.num_nonlinear_pixels //= 2
            stats.num_susceptibility_pixels //= 2
            stats.num_nonzero_conductivity_pixels //= 2
            stats.num_1d_pml_pixels //= 2
            stats.num_2d_pml_pixels //= 2
            stats.num_3d_pml_pixels //= 2
            stats.num_pixels_in_box //= 2

        return stats

    def _init_structure(self, k=False):
        if verbosity.meep > 0:
            print('-' * 11)
            print('Initializing structure...')

        gv = self._create_grid_volume(k)
        sym = self._create_symmetries(gv)
        br = _create_boundary_region_from_boundary_layers(self.boundary_layers, gv)
        absorbers = [bl for bl in self.boundary_layers if type(bl) is Absorber]

        if self.material_function:
            init_do_averaging(self.material_function)
            self.material_function.eps = False
            self.default_material = self.material_function
        elif self.epsilon_func:
            init_do_averaging(self.epsilon_func)
            self.epsilon_func.eps = True
            self.default_material = self.epsilon_func
        elif self.epsilon_input_file:
            self.default_material = self.epsilon_input_file

        if self.collect_stats and isinstance(self.default_material, mp.Medium):
            self.fragment_stats = self._compute_fragment_stats(gv)

        if self._output_stats and isinstance(self.default_material, mp.Medium) and verbosity.meep > 0:
            stats = self._compute_fragment_stats(gv)
            print("FRAGMENT:, aniso_eps:, {}".format(stats.num_anisotropic_eps_pixels))
            print("FRAGMENT:, aniso_mu:, {}".format(stats.num_anisotropic_mu_pixels))
            print("FRAGMENT:, nonlinear:, {}".format(stats.num_nonlinear_pixels))
            print("FRAGMENT:, susceptibility:, {}".format(stats.num_susceptibility_pixels))
            print("FRAGMENT:, conductivity:, {}".format(stats.num_nonzero_conductivity_pixels))
            print("FRAGMENT:, pml_1d:, {}".format(stats.num_1d_pml_pixels))
            print("FRAGMENT:, pml_2d:, {}".format(stats.num_2d_pml_pixels))
            print("FRAGMENT:, pml_3d:, {}".format(stats.num_3d_pml_pixels))
            print("FRAGMENT:, dft:, {}".format(stats.num_dft_pixels))
            print("FRAGMENT:, total_pixels:, {}".format(stats.num_pixels_in_box))
            print("FRAGMENT:, procs:, {}".format(mp.count_processors()))

        fragment_vols = self._make_fragment_lists(gv)
        self.dft_data_list = fragment_vols[0]
        self.pml_vols1 = fragment_vols[1]
        self.pml_vols2 = fragment_vols[2]
        self.pml_vols3 = fragment_vols[3]
        self.absorber_vols = fragment_vols[4]
        self.gv = gv

        self.structure = mp.create_structure_and_set_materials(
            self.cell_size,
            self.dft_data_list,
            self.pml_vols1,
            self.pml_vols2,
            self.pml_vols3,
            self.absorber_vols,
            gv,
            br,
            sym,
            self.num_chunks,
            self.Courant,
            self.eps_averaging,
            self.subpixel_tol,
            self.subpixel_maxeval,
            self.geometry,
            self.geometry_center,
            self.ensure_periodicity and not not self.k_point,
            self.default_material,
            absorbers,
            self.extra_materials,
            self.split_chunks_evenly,
            False if self.chunk_layout and not isinstance(self.chunk_layout,mp.BinaryPartition) else True,
            None,
            True if self._output_stats is not None else False,
            self.chunk_layout if self.chunk_layout and isinstance(self.chunk_layout,mp.BinaryPartition) else None
        )

        if self._output_stats is not None:
            sys.exit(0)

        if self.chunk_layout and not isinstance(self.chunk_layout,mp.BinaryPartition):
            self.load_chunk_layout(br, self.chunk_layout)
            self.set_materials()

        if self.load_structure_file:
            self.load_structure(self.load_structure_file)

    def _is_outer_boundary(self, vol, direction, side):

        if direction == mp.X:
            cell_size_in_dir = self.cell_size.x
        elif direction == mp.Y:
            cell_size_in_dir = self.cell_size.y
        else:
            cell_size_in_dir = self.cell_size.z

        half_cell_size = cell_size_in_dir / 2
        # TODO: Support shifted origins

        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

        if side == mp.Low and isclose(vol.get_min_corner().in_direction(direction), -half_cell_size):
            return True
        if side == mp.High and isclose(vol.get_max_corner().in_direction(direction), half_cell_size):
            return True

        return False

    def _get_chunk_communication_area(self, vol):

        result = 0

        def get_num_pixels(vol, direction, side, target_direction, mult_direction=None):
            result = 0
            if not self._is_outer_boundary(vol, direction, side):
                result = vol.in_direction(target_direction)
                if mult_direction is not None:
                    result *= vol.in_direction(mult_direction)
            else:
                # Check for periodic outer boundary
                if self.fields.is_periodic(side, direction):
                    result = vol.in_direction(target_direction)
                    if mult_direction is not None:
                        result *= vol.in_direction(mult_direction)
            return result

        if vol.dim == 1:
            # 2d
            yLow = get_num_pixels(vol, mp.X, mp.Low, mp.Y)
            yHigh = get_num_pixels(vol, mp.X, mp.High, mp.Y)
            xLow = get_num_pixels(vol, mp.Y, mp.Low, mp.X)
            xHigh = get_num_pixels(vol, mp.Y, mp.High, mp.X)
            result = xLow + xHigh + yLow + yHigh
        else:
            # 3d
            yLow = get_num_pixels(vol, mp.X, mp.Low, mp.Y, mp.Z)
            yHigh = get_num_pixels(vol, mp.X, mp.High, mp.Y, mp.Z)
            xLow = get_num_pixels(vol, mp.Z, mp.Low, mp.Y, mp.X)
            xHigh = get_num_pixels(vol, mp.Z, mp.High, mp.Y, mp.X)
            zLow = get_num_pixels(vol, mp.Y, mp.Low, mp.X, mp.Z)
            zHigh = get_num_pixels(vol, mp.Y, mp.Low, mp.X, mp.Z)
            result = yLow + yHigh + xLow + xHigh + zLow + zHigh

        return result * self.resolution

    def _get_chunk_communication_areas(self):

        if self.dimensions == 1 or self.is_cylindrical:
            warnings.warn("Can currently only get chunk communication area from 2d or 3d simulations",
                          RuntimeWarning)
            return

        if self.structure is None:
            self.init_sim()

        vols = self.structure.get_chunk_volumes()
        owners = self.structure.get_chunk_owners()

        # Union the chunk volumes that are on the same processor
        idx = 0
        result = []
        for i in range(mp.count_processors()):
            unioned_vol = vols[idx].surroundings()
            idx += 1
            while idx < len(owners) and owners[idx] == i:
                unioned_vol = unioned_vol | vols[idx].surroundings()
                idx += 1
            result.append(self._get_chunk_communication_area(unioned_vol))

        return result

    def get_max_chunk_communication_area(self):
        return max(self._get_chunk_communication_areas())

    def get_avg_chunk_communication_area(self):
        return sum(self._get_chunk_communication_areas()) / mp.count_processors()

    def get_estimated_costs(self):
        return [self.structure.estimated_cost(i) for i in range(mp.count_processors())]

    def set_materials(self, geometry=None, default_material=None):
        """
        This can be called in a step function, and is useful for changing the geometry or
        default material as a function of time.
        """
        if self.fields:
            self.fields.remove_susceptibilities()

        absorbers = [bl for bl in self.boundary_layers if type(bl) is Absorber]

        # Since we are about to overwrite self.structure, SWIG will garbage
        # collect it. However, that's not what we want because we're just
        # passing self.structure into create_structure_and_set_materials for
        # the "set_materials" half of that function. The return value will be
        # the same structure we passed in. We tell SWIG to disown (and not
        # delete) the current self.structure. SWIG will properly take ownership
        # of the returned self.structure (which is the same structure as
        # before).
        self.structure.this.disown()

        self.structure = mp.create_structure_and_set_materials(
            self.cell_size,
            self.dft_data_list,
            self.pml_vols1,
            self.pml_vols2,
            self.pml_vols3,
            self.absorber_vols,
            self.gv,
            mp.boundary_region(),
            mp.symmetry(),
            self.num_chunks,
            self.Courant,
            self.eps_averaging,
            self.subpixel_tol,
            self.subpixel_maxeval,
            geometry if geometry is not None else self.geometry,
            self.geometry_center,
            self.ensure_periodicity and not not self.k_point,
            default_material if default_material else self.default_material,
            absorbers,
            self.extra_materials,
            self.split_chunks_evenly,
            True,
            self.structure,
            False,
            None
        )

    def dump_structure(self, fname):
        """
        Dumps the structure to the file `fname`.
        """
        if self.structure is None:
            raise ValueError("Fields must be initialized before calling dump_structure")
        self.structure.dump(fname)

    def load_structure(self, fname):
        """
        Loads a structure from the file `fname`. A file name to load can also be passed to
        the `Simulation` constructor via the `load_structure` keyword argument.
        """
        if self.structure is None:
            raise ValueError("Fields must be initialized before calling load_structure")
        self.structure.load(fname)

    def dump_chunk_layout(self, fname):
        """
        Dumps the chunk layout to file `fname`.
        """
        if self.structure is None:
            raise ValueError("Fields must be initialized before calling load_structure")
        self.structure.dump_chunk_layout(fname)

    def load_chunk_layout(self, br, source):
        if self.structure is None:
            raise ValueError("Fields must be initialized before calling load_structure")

        if isinstance(source, Simulation):
            vols = source.structure.get_chunk_volumes()
            ids = source.structure.get_chunk_owners()
            self.structure.load_chunk_layout(vols, [int(f) for f in ids], br)
        else:
            ## source is either filename (string)
            self.structure.load_chunk_layout(source, br)

    def init_sim(self):
        if self._is_initialized:
            return

        materials = [g.material for g in self.geometry if isinstance(g.material, mp.Medium)]
        if isinstance(self.default_material, mp.Medium):
            materials.append(self.default_material)
        for med in materials:
            if ((med.epsilon_diag.x < 1 and med.epsilon_diag.x > -mp.inf) or
                (med.epsilon_diag.y < 1 and med.epsilon_diag.y > -mp.inf) or
                (med.epsilon_diag.z < 1 and med.epsilon_diag.z > -mp.inf)):

                eps_warning = ("Epsilon < 1 may require adjusting the Courant parameter. " +
                               "See the 'Numerical Stability' entry under the 'Materials' " +
                               "section of the documentation")
                warnings.warn(eps_warning, RuntimeWarning)

        if self.structure is None:
            self._init_structure(self.k_point)

        self.fields = mp.fields(
            self.structure,
            self.m if self.is_cylindrical else 0,
            self.k_point.z if self.special_kz and self.k_point else 0,
            not self.accurate_fields_near_cylorigin
        )

        if self.force_all_components and self.dimensions != 1:
            self.fields.require_component(mp.Ez)
            self.fields.require_component(mp.Hz)

        def use_real(self):
            cond1 = self.is_cylindrical and self.m != 0
            cond2 = any([s.phase.imag for s in self.symmetries])
            cond3 = not self.k_point
            cond4 = self.special_kz and self.k_point.x == 0 and self.k_point.y == 0
            cond5 = not (cond3 or cond4 or self.k_point == Vector3())
            return not (self.force_complex_fields or cond1 or cond2 or cond5)

        if use_real(self):
            self.fields.use_real_fields()
        elif verbosity.meep > 0:
            print("Meep: using complex fields.")

        if self.k_point:
            v = Vector3(self.k_point.x, self.k_point.y) if self.special_kz else self.k_point
            self.fields.use_bloch(py_v3_to_vec(self.dimensions, v, self.is_cylindrical))

        self.add_sources()

        for hook in self.init_sim_hooks:
            hook()

        self._is_initialized = True

    def initialize_field(self, cmpnt, amp_func):
        """
        Initialize the component `c` fields using the function `func` which has a single
        argument, a `Vector3` giving a position and returns a complex number for the value
        of the field at that point.
        """
        if self.fields is None:
            self.init_sim()
        self.fields.initialize_field(cmpnt, amp_func)

    def require_dimensions(self):
        if self.structure is None:
            mp.set_dimensions(self._infer_dimensions(self.k_point))

    def has_mu(self):

        def _has_mu(medium):
            if not isinstance(medium, mp.Medium):
                return False
            return medium.mu_diag != mp.Vector3(1, 1, 1) or medium.mu_offdiag != mp.Vector3(0j, 0j, 0j)

        for go in self.geometry:
            if _has_mu(go.material):
                return True

        for mat in self.extra_materials:
            if _has_mu(mat):
                return True

        return _has_mu(self.default_material)

    def get_estimated_memory_usage(self):
        if self.fields is None:
            self.collect_stats = True
            self.init_sim()

        if self.fragment_stats is None:
            self.fragment_stats = self._compute_fragment_stats(self.structure.user_volume)

        is_complex = (self.k_point and self.k_point != mp.Vector3(0, 0, 0)) or self.force_complex_fields
        realnums_per_grid_point = 1 if self.dimensions == 1 else 3
        E_realnums = self.fragment_stats.num_pixels_in_box * (2 if is_complex else 1) * realnums_per_grid_point
        H_realnums = self.fragment_stats.num_pixels_in_box * (2 if is_complex else 1) * realnums_per_grid_point
        D_realnums = self.fragment_stats.num_pixels_in_box * (2 if is_complex else 1) * realnums_per_grid_point
        chi1inv_realnums = self.fragment_stats.num_pixels_in_box * 9

        Mu_realnums = 0
        if self.has_mu():
            Mu_realnums = chi1inv_realnums + H_realnums

        dft_realnums = self.fragment_stats.num_dft_pixels * 2
        dispersive_realnums = self.fragment_stats.num_susceptibility_pixels * 6 * (2 if is_complex else 1)

        total_realnums = (E_realnums + H_realnums + D_realnums + Mu_realnums +
                          dft_realnums + dispersive_realnums)

        total_bytes = total_realnums * mp.get_realnum_size()

        return total_bytes

    def meep_time(self):
        """
        Return the current simulation time in simulation time units (e.g. during a run
        function). This is not the wall-clock time.

        Occasionally, e.g. for termination conditions of the form $time < T?$, it is
        desirable to round the time to single precision in order to avoid small
        differences in roundoff error from making your results different by one timestep
        from machine to machine (a difference much bigger than roundoff error); in this
        case you can call `Simulation.round_time()` instead, which returns the time
        rounded to single precision.
        """
        if self.fields is None:
            self.init_sim()
        return self.fields.time()

    def round_time(self):
        if self.fields is None:
            self.init_sim()

        return self.fields.round_time()

    def phase_in_material(self, structure, time):
        """
        `newstructure` should be the `structure` field of another
        `Simulation` object with the same cell size and resolution.
        Over the next time period `phasetime` (in the current
        simulation's time units), the current structure
        ($\\varepsilon$, $\\mu$, and conductivity $\\sigma_D$) will be
        gradually changed to `newstructure`. In particular, at each
        timestep it linearly interpolates between the old structure
        and the new structure. After `phasetime` has elapsed, the
        structure will remain equal to `newstructure`. This is
        demonstrated in the following image for two
        [Cylinder](#cylinder) objects (the simulation script is in
        [examples/phase_in_material.py](https://github.com/NanoComp/meep/blob/master/python/examples/phase_in_material.py)).

        ![](images/phase-in-material.png)
        """
        if self.fields is None:
            self.init_sim()

        return self.fields.phase_in_material(structure, time)

    def set_boundary(self, side, direction, condition):
        """
        Sets the condition of the boundary on the specified side in the specified
        direction. See the [Constants (Enumerated Types)](#constants-enumerated-types)
        section for valid `side`, `direction`, and `boundary_condition` values.
        """
        if self.fields is None:
            self.init_sim()

        self.fields.set_boundary(side, direction, condition)

    def get_field_point(self, c, pt):
        """
        Given a `component` or `derived_component` constant `c` and a `Vector3` `pt`,
        returns the value of that component at that point.
        """
        v3 = py_v3_to_vec(self.dimensions, pt, self.is_cylindrical)
        return self.fields.get_field_from_comp(c, v3)

    def get_epsilon_point(self, pt, frequency=0):
        """
        Given a frequency `frequency` and a `Vector3` `pt`, returns the average eigenvalue
        of the permittivity tensor at that location and frequency. If `frequency` is
        non-zero, the result is complex valued; otherwise it is the real,
        frequency-independent part of $\\varepsilon$ (the $\\omega\\to\\infty$ limit).
        """
        v3 = py_v3_to_vec(self.dimensions, pt, self.is_cylindrical)
        return self.fields.get_eps(v3,frequency)

    def get_mu_point(self, pt, frequency=0):
        """
        Given a frequency `frequency` and a `Vector3` `pt`, returns the average eigenvalue
        of the permeability tensor at that location and frequency. If `frequency` is
        non-zero, the result is complex valued; otherwise it is the real,
        frequency-independent part of $\\mu$ (the $\\omega\\to\\infty$ limit).
        """
        v3 = py_v3_to_vec(self.dimensions, pt, self.is_cylindrical)
        return self.fields.get_mu(v3,frequency)

    def get_epsilon_grid(self, xtics=None, ytics=None, ztics=None):
        """
        Given three 1d NumPy arrays (`xtics`,`ytics`,`ztics`) which define the coordinates of a Cartesian
        grid anywhere within the cell volume, compute the trace of the $\\varepsilon$ tensor from the `geometry`
        exactly at each grid point. (For [`MaterialGrid`](#materialgrid)s, the $\\varepsilon$ at each grid
        point is computed using bilinear interpolation from the nearest `MaterialGrid` points and possibly also
        projected to form a level set.) Note that this is different from `get_epsilon_point` which computes
        $\\varepsilon$ by bilinearly interpolating from the nearest Yee grid points. This function is useful for
        sampling the material geometry to any arbitrary resolution. The return value is a NumPy array with shape
        equivalent to `numpy.meshgrid(xtics,ytics,ztics)`. Empty dimensions are collapsed.
        """
        grid_vals = np.squeeze(np.empty((len(xtics), len(ytics), len(ztics))))
        gv = self._create_grid_volume(False)
        mp._get_epsilon_grid(self.geometry,
                             self.extra_materials,
                             self.default_material,
                             self.ensure_periodicity and not not self.k_point,
                             gv,
                             self.cell_size, self.geometry_center,
                             len(xtics), xtics,
                             len(ytics), ytics,
                             len(ztics), ztics,
                             grid_vals)
        return grid_vals

    def get_filename_prefix(self):
        """
        Return the current prefix string that is prepended, by default, to all file names.

        If you don't want to use any prefix, then you should set `filename_prefix` to the
        empty string `''`.

        In addition to the filename prefix, you can also specify that all the output files
        be written into a newly-created directory (if it does not yet exist). This is done
        by calling `Simulation.use_output_directory([dirname])`
        """
        if isinstance(self.filename_prefix, str):
            return self.filename_prefix
        elif self.filename_prefix is None:
            _, filename = os.path.split(sys.argv[0])

            if filename == 'ipykernel_launcher.py' or filename == '__main__.py':
                return ''
            else:
                return re.sub(r'\.py$', '', filename)
        else:
            raise TypeError("Expected a string for filename_prefix, or None for the default.")

    def use_output_directory(self, dname=''):
        """
        Output all files into a subdirectory, which is created if necessary. If the optional
        argument `dname` is specified, that is the name of the directory. If `dname`
        is omitted and `filename_prefix` is `None`, the directory name is the current Python
        filename with `".py"` replaced by `"-out"`: e.g. `test.py` implies a directory of
        `"test-out"`. If `dname` is omitted and `filename_prefix` has been set, the directory
        name is set to `filename_prefix` + "-out" and `filename_prefix` is then reset to `None`.
        """
        if not dname:
            dname = self.get_filename_prefix() + '-out'

        closure = {'trashed': False}

        def hook():
            if verbosity.meep > 0:
                print("Meep: using output directory '{}'".format(dname))
            self.fields.set_output_directory(dname)
            if not closure['trashed']:
                mp.trash_output_directory(dname)
            closure['trashed'] = True

        self.init_sim_hooks.append(hook)

        if self.fields is not None:
            hook()
        self.filename_prefix = None

        return dname

    def _run_until(self, cond, step_funcs):
        self.interactive = False
        if self.fields is None:
            self.init_sim()

        if not isinstance(cond, list):
            cond = [cond]

        self.progress = False
        for i in range(len(cond)):
            if isinstance(cond[i], numbers.Number):
                stop_time = cond[i]
                t0 = self.round_time()

                def stop_cond(sim):
                    return sim.round_time() >= t0 + stop_time

                cond[i] = stop_cond

                step_funcs = list(step_funcs)
                step_funcs.append(display_progress(t0, t0 + stop_time, self.progress_interval))

                if do_progress:
                    self.progress = FloatProgress(value=t0, min=t0, max=t0 + stop_time, description="0% done ")
                    display(self.progress)
            else:
                assert callable(cond[i]), "Stopping condition {} is not an integer or a function".format(cond[i])

        while not any([x(self) for x in cond]):
            for func in step_funcs:
                _eval_step_func(self, func, 'step')
            self.fields.step()

        # Translating the recursive scheme version of run-until into an iterative version
        # (because python isn't tail-call-optimized) means we need one extra iteration to
        # be the same as scheme.
        for func in step_funcs:
            _eval_step_func(self, func, 'step')

        for func in step_funcs:
            _eval_step_func(self, func, 'finish')

        if do_progress and self.progress:
            self.progress.value = t0 + stop_time
            self.progress.description = "100% done "

        if verbosity.meep > 0:
            print("run {} finished at t = {} ({} timesteps)".format(self.run_index, self.meep_time(), self.fields.t))
        self.run_index += 1

    def _run_sources_until(self, cond, step_funcs):
        if self.fields is None:
            self.init_sim()

        if not isinstance(cond, list):
            cond = [cond]

        ts = self.fields.last_source_time()
        new_conds = []
        for i in range(len(cond)):
            if isinstance(cond[i], numbers.Number):
                new_conds.append((ts - self.round_time()) + cond[i])
            else:
                def f(sim):
                    return cond[i](sim) and sim.round_time() >= ts
                new_conds.append(f)

        self._run_until(new_conds, step_funcs)

    def _run_sources(self, step_funcs):
        """
        Lower level function called by `run_k_points` that runs a simulation for a single
        *k* point `k_point` and returns a `Harminv` instance. Useful when you need to
        access more `Harminv` data than just the frequencies.
        """
        self._run_sources_until(self, 0, step_funcs)

    def run_k_point(self, t, k):
        """
        Lower level function called by `run_k_points` that runs a simulation for a single
        *k* point `k_point` and returns a `Harminv` instance. Useful when you need to
        access more `Harminv` data than just the frequencies.
        """
        components = [s.component for s in self.sources]
        pts = [s.center for s in self.sources]

        src_freqs_min = min([s.src.frequency - 1 / s.src.width / 2 if isinstance(s.src, mp.GaussianSource) else mp.inf
                             for s in self.sources])
        fmin = max(0, src_freqs_min)

        fmax = max([s.src.frequency + 1 / s.src.width / 2 if isinstance(s.src, mp.GaussianSource) else 0
                    for s in self.sources])

        if not components or fmin > fmax:
            raise ValueError("Running with k_points requires a 'GaussianSource' source")

        self.change_k_point(k)
        self.restart_fields()

        h = Harminv(components[0], pts[0], 0.5 * (fmin + fmax), fmax - fmin)
        self.run(after_sources(h), until_after_sources=t)

        return h

    def run_k_points(self, t, k_points):
        """
        Given a list of `Vector3`, `k_points` of *k* vectors, runs a simulation for each
        *k* point (i.e. specifying Bloch-periodic boundary conditions) and extracts the
        eigen-frequencies, and returns a list of the complex frequencies. In particular,
        you should have specified one or more Gaussian sources. It will run the simulation
        until the sources are turned off plus an additional $t$ time units. It will run
        [Harminv](#harminv) at the same point/component as the first Gaussian source and
        look for modes in the union of the frequency ranges for all sources. Returns a
        list of lists of frequencies (one list of frequencies for each *k*). Also prints
        out a comma-delimited list of frequencies, prefixed by `freqs:`, and their
        imaginary parts, prefixed by `freqs-im:`. See [Tutorial/Resonant Modes and
        Transmission in a Waveguide
        Cavity](Python_Tutorials/Resonant_Modes_and_Transmission_in_a_Waveguide_Cavity.md).
        """
        k_index = 0
        all_freqs = []

        for k in k_points:
            k_index += 1
            harminv = self.run_k_point(t, k)
            freqs = [complex(m.freq, m.decay) for m in harminv.modes]

            print("freqs:, {}, {}, {}, {}, ".format(k_index, k.x, k.y, k.z), end='')
            print(', '.join([str(f.real) for f in freqs]))
            print("freqs-im:, {}, {}, {}, {}, ".format(k_index, k.x, k.y, k.z), end='')
            print(', '.join([str(f.imag) for f in freqs]))

            all_freqs.append(freqs)

        return all_freqs

    def set_epsilon(self, eps):
        if self.fields is None:
            self.init_sim()

        self.structure.set_epsilon(eps, self.eps_averaging, self.subpixel_tol, self.subpixel_maxeval)

    def add_source(self, src):
        if self.fields is None:
            self.init_sim()

        if isinstance(src, IndexedSource):
            self.fields.add_srcdata(src.srcdata, src.src.swigobj, src.num_pts, src.amp_arr)
            return

        where = Volume(src.center, src.size, dims=self.dimensions,
                       is_cylindrical=self.is_cylindrical).swigobj

        if isinstance(src, EigenModeSource):
            if src.direction < 0:
                direction = self.fields.normal_direction(where)
            else:
                direction = src.direction

            eig_vol = Volume(src.eig_lattice_center, src.eig_lattice_size, self.dimensions,
                             is_cylindrical=self.is_cylindrical).swigobj

            if isinstance(src.eig_band, DiffractedPlanewave):
                eig_band = 1
                diffractedplanewave = bands_to_diffractedplanewave(where, src.eig_band)
            elif isinstance(src.eig_band, int):
                eig_band = src.eig_band

            add_eig_src_args = [
                src.component,
                src.src.swigobj,
                direction,
                where,
                eig_vol,
                eig_band,
                py_v3_to_vec(self.dimensions, src.eig_kpoint, is_cylindrical=self.is_cylindrical),
                src.eig_match_freq,
                src.eig_parity,
                src.eig_resolution,
                src.eig_tolerance,
                src.amplitude
            ]
            add_eig_src = functools.partial(self.fields.add_eigenmode_source, *add_eig_src_args)

            if isinstance(src.eig_band, DiffractedPlanewave):
                add_eig_src(src.amp_func, diffractedplanewave)
            else:
                add_eig_src(src.amp_func)
        elif isinstance (src, GaussianBeamSource):
            gaussianbeam_args = [
                py_v3_to_vec(self.dimensions, src.beam_x0, is_cylindrical=self.is_cylindrical),
                py_v3_to_vec(self.dimensions, src.beam_kdir, is_cylindrical=self.is_cylindrical),
                src.beam_w0,
                src.src.swigobj.frequency().real,
                self.fields.get_eps(py_v3_to_vec(self.dimensions, src.center, self.is_cylindrical)).real,
                self.fields.get_mu(py_v3_to_vec(self.dimensions, src.center, self.is_cylindrical)).real,
                np.array([src.beam_E0.x, src.beam_E0.y, src.beam_E0.z], dtype=np.complex128)
            ]
            gaussianbeam = mp.gaussianbeam(*gaussianbeam_args)
            add_vol_src_args = [src.src.swigobj, where, gaussianbeam]
            add_vol_src = functools.partial(self.fields.add_volume_source, *add_vol_src_args)
            add_vol_src()
        else:
            add_vol_src_args = [src.component, src.src.swigobj, where]
            add_vol_src = functools.partial(self.fields.add_volume_source, *add_vol_src_args)

            if src.amp_func_file:
                fname_dset = src.amp_func_file.rsplit(':', 1)
                if len(fname_dset) != 2:
                    err_msg = "Expected a string of the form 'h5filename:dataset'. Got '{}'"
                    raise ValueError(err_msg.format(src.amp_func_file))

                fname, dset = fname_dset
                if not fname.endswith('.h5'):
                    fname += '.h5'

                add_vol_src(fname, dset, src.amplitude * 1.0)
            elif src.amp_func:
                add_vol_src(src.amp_func, src.amplitude * 1.0)
            elif src.amp_data is not None:
                add_vol_src(src.amp_data, src.amplitude * 1.0)
            else:
                add_vol_src(src.amplitude * 1.0)

    def add_sources(self):
        if self.fields is None:
            self.init_sim() # in case only some processes have IndexedSources
        for s in self.sources:
            self.add_source(s)
        self.fields.require_source_components() # needed by IndexedSource objects

    def _evaluate_dft_objects(self):
        for dft in self.dft_objects:
            if dft.swigobj is None:
                dft.swigobj = dft.func(*dft.args)

    def add_dft_fields(self, *args, **kwargs):
        """
        `add_dft_fields(cs, fcen, df, nfreq, freq, where=None, center=None, size=None, yee_grid=False)` ##sig

        Given a list of field components `cs`, compute the Fourier transform of these
        fields for `nfreq` equally spaced frequencies covering the frequency range
        `fcen-df/2` to `fcen+df/2` or an array/list `freq` for arbitrarily spaced
        frequencies over the `Volume` specified by `where` (default to the entire cell).
        The volume can also be specified via the `center` and `size` arguments. The
        default routine interpolates the Fourier transformed fields at the center of each
        voxel within the specified volume. Alternatively, the exact Fourier transformed
        fields evaluated at each corresponding Yee grid point is available by setting
        `yee_grid` to `True`.
        """
        components = args[0]
        args = fix_dft_args(args, 1)
        freq = args[1]
        where = kwargs.get('where', None)
        center = kwargs.get('center', None)
        size = kwargs.get('size', None)
        yee_grid = kwargs.get('yee_grid', False)
        center_v3 = Vector3(*center) if center is not None else None
        size_v3 = Vector3(*size) if size is not None else None
        use_centered_grid = not yee_grid
        dftf = DftFields(self._add_dft_fields, [components, where, center_v3, size_v3, freq, use_centered_grid])
        self.dft_objects.append(dftf)
        return dftf

    def _add_dft_fields(self, components, where, center, size, freq, use_centered_grid):
        if self.fields is None:
            self.init_sim()
        try:
            where = self._volume_from_kwargs(where, center, size)
        except ValueError:
            where = self.fields.total_volume()
        return self.fields.add_dft_fields(components, where, freq, use_centered_grid)

    def output_dft(self, dft_fields, fname):
        """
        Output the Fourier-transformed fields in `dft_fields` (created by
        `add_dft_fields`) to an HDF5 file with name `fname` (does *not* include the `.h5`
        suffix).
        """
        if self.fields is None:
            self.init_sim()

        if not self.dft_objects:
            raise RuntimeError('DFT monitor dft_fields must be initialized before calling output_dft')

        if hasattr(dft_fields, 'swigobj'):
            dft_fields_swigobj = dft_fields.swigobj
        else:
            dft_fields_swigobj = dft_fields

        self.fields.output_dft(dft_fields_swigobj, fname)

    def get_dft_data(self, dft_chunk):
        n = mp._get_dft_data_size(dft_chunk)
        arr = np.zeros(n, np.complex128)
        mp._get_dft_data(dft_chunk, arr)
        return arr

    def add_near2far(self, *args, **kwargs):
        """
        `add_near2far(fcen, df, nfreq, freq, Near2FarRegions..., nperiods=1)`  ##sig

        Add a bunch of `Near2FarRegion`s to the current simulation (initializing the
        fields if they have not yet been initialized), telling Meep to accumulate the
        appropriate field Fourier transforms for `nfreq` equally spaced frequencies
        covering the frequency range `fcen-df/2` to `fcen+df/2` or an array/list `freq`
        for arbitrarily spaced frequencies. Return a `near2far` object, which you can pass
        to the functions below to get the far fields.
        """
        args = fix_dft_args(args, 0)
        freq = args[0]
        near2fars = args[1:]
        nperiods = kwargs.get('nperiods', 1)
        n2f = DftNear2Far(self._add_near2far, [freq, nperiods, near2fars])
        self.dft_objects.append(n2f)
        return n2f

    def _add_near2far(self, freq, nperiods, near2fars):
        if self.fields is None:
            self.init_sim()
        return self._add_fluxish_stuff(self.fields.add_dft_near2far, freq, near2fars, nperiods)

    def add_energy(self, *args):
        """
        `add_energy(fcen, df, nfreq, freq, EnergyRegions...)`  ##sig

        Add a bunch of `EnergyRegion`s to the current simulation (initializing the fields
        if they have not yet been initialized), telling Meep to accumulate the appropriate
        field Fourier transforms for `nfreq` equally spaced frequencies covering the
        frequency range `fcen-df/2` to `fcen+df/2` or an array/list `freq` for arbitrarily
        spaced frequencies. Return an *energy object*, which you can pass to the functions
        below to get the energy spectrum, etcetera.
        """
        args = fix_dft_args(args, 0)
        freq = args[0]
        energys = args[1:]
        en = DftEnergy(self._add_energy, [freq, energys])
        self.dft_objects.append(en)
        return en

    def _add_energy(self, freq, energys):
        if self.fields is None:
            self.init_sim()
        return self._add_fluxish_stuff(self.fields.add_dft_energy, freq, energys)

    def _display_energy(self, name, func, energys):
        if energys:
            freqs = get_energy_freqs(energys[0])
            display_csv(self, "{}-energy".format(name), zip(freqs, *[func(f) for f in energys]))

    def display_electric_energy(self, *energys):
        """
        Given a number of energy objects, this displays a comma-separated table of
        frequencies and energy density spectra for the electric fields prefixed by
        "electric_energy1:" or similar (where the number is incremented after each run).
        All of the energy should be for the same `fcen`/`df`/`nfreq` or `freq`. The first
        column are the frequencies, and subsequent columns are the energy density spectra.
        """
        self._display_energy('electric', get_electric_energy, energys)

    def display_magnetic_energy(self, *energys):
        """
        Given a number of energy objects, this displays a comma-separated table of
        frequencies and energy density spectra for the magnetic fields prefixed by
        "magnetic_energy1:" or similar (where the number is incremented after each run).
        All of the energy should be for the same `fcen`/`df`/`nfreq` or `freq`. The first
        column are the frequencies, and subsequent columns are the energy density spectra.
        """
        self._display_energy('magnetic', get_magnetic_energy, energys)

    def display_total_energy(self, *energys):
        """
        Given a number of energy objects, this displays a comma-separated table of
        frequencies and energy density spectra for the total fields "total_energy1:" or
        similar (where the number is incremented after each run). All of the energy should
        be for the same `fcen`/`df`/`nfreq` or `freq`. The first column are the
        frequencies, and subsequent columns are the energy density spectra.
        """
        self._display_energy('total', get_total_energy, energys)

    def load_energy(self, fname, energy):
        """
        Load the Fourier-transformed fields into the given energy object (replacing any
        values currently there) from an HDF5 file of the given `filename` without the
        `.h5` suffix (the current filename-prefix is prepended automatically). You must
        load from a file that was saved by `save_energy` in a simulation of the same
        dimensions for both the cell and the energy regions with the same number of
        processors.
        """
        if self.fields is None:
            self.init_sim()
        energy.load_hdf5(self.fields, fname, '', self.get_filename_prefix())

    def save_energy(self, fname, energy):
        """
        Save the Fourier-transformed fields corresponding to the given energy object in an
        HDF5 file of the given `filename` without the `.h5` suffix (the current
        filename-prefix is prepended automatically).
        """
        if self.fields is None:
            self.init_sim()
        energy.save_hdf5(self.fields, fname, '', self.get_filename_prefix())

    def load_minus_energy(self, fname, energy):
        """
        As `load_energy`, but negates the Fourier-transformed fields after they are
        loaded. This means that they will be *subtracted* from any future field Fourier
        transforms that are accumulated.
        """
        self.load_energy(fname, energy)
        energy.scale_dfts(-1.0)

    def get_farfield(self, near2far, x):
        """
        Given a `Vector3` point `x` which can lie anywhere outside the near-field surface,
        including outside the cell and a `near2far` object, returns the computed
        (Fourier-transformed) "far" fields at `x` as list of length 6`nfreq`, consisting
        of fields $(E_x^1,E_y^1,E_z^1,H_x^1,H_y^1,H_z^1,E_x^2,E_y^2,E_z^2,H_x^2,H_y^2,H_z^2,...)$
        for the frequencies 1,2,…,`nfreq`.
        """
        return mp._get_farfield(near2far.swigobj, py_v3_to_vec(self.dimensions, x, is_cylindrical=self.is_cylindrical))

    def get_farfields(self, near2far, resolution, where=None, center=None, size=None):
        """
        Like `output_farfields` but returns a dictionary of NumPy arrays instead of
        writing to a file. The dictionary keys are `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, `Hz`.
        Each array has the same shape as described in `output_farfields`.

        Note that far fields have the same units and scaling as the *Fourier transforms*
        of the fields, and hence cannot be directly compared to time-domain fields. In
        practice, it is easiest to use the far fields in computations where overall
        scaling (units) cancel out or are irrelevant, e.g. to compute the fraction of the
        far fields in one region vs. another region.
        """
        if self.fields is None:
            self.init_sim()
        vol = self._volume_from_kwargs(where, center, size)
        self.fields.am_now_working_on(mp.GetFarfieldsTime)
        result = mp._get_farfields_array(near2far.swigobj, vol, resolution)
        self.fields.finished_working()
        res_ex = complexarray(result[0], result[1])
        res_ey = complexarray(result[2], result[3])
        res_ez = complexarray(result[4], result[5])
        res_hx = complexarray(result[6], result[7])
        res_hy = complexarray(result[8], result[9])
        res_hz = complexarray(result[10], result[11])
        return {
            'Ex': res_ex,
            'Ey': res_ey,
            'Ez': res_ez,
            'Hx': res_hx,
            'Hy': res_hy,
            'Hz': res_hz,
        }

    def output_farfields(self, near2far, fname, resolution, where=None, center=None, size=None):
        """
        Given an HDF5 file name `fname` (does *not* include the `.h5` suffix), a `Volume`
        given by `where` (may be 0d, 1d, 2d, or 3d), and a `resolution` (in grid points /
        distance unit), outputs the far fields in `where` (which may lie *outside* the
        cell) in a grid with the given resolution (which may differ from the FDTD grid
        resolution) to the HDF5 file as a set of twelve array datasets `ex.r`, `ex.i`,
        ..., `hz.r`, `hz.i`, giving the real and imaginary parts of the
        Fourier-transformed $E$ and $H$ fields on this grid. Each dataset is an
        nx&#215;ny&#215;nz&#215;nfreq 4d array of space&#215;frequency although dimensions
        that =1 are omitted. The volume can optionally be specified via `center` and
        `size`.
        """
        if self.fields is None:
            self.init_sim()
        vol = self._volume_from_kwargs(where, center, size)
        self.fields.am_now_working_on(mp.GetFarfieldsTime)
        near2far.save_farfields(fname, self.get_filename_prefix(), vol, resolution)
        self.fields.finished_working()

    def load_near2far(self, fname, near2far):
        """
        Load the Fourier-transformed fields into the given `near2far` object (replacing
        any values currently there) from an HDF5 file of the given `filename` without the
        `.h5` suffix (the current filename-prefix is prepended automatically). You must
        load from a file that was saved by `save_near2far` in a simulation of *the same
        dimensions* for both the cell and the near2far regions with the same number of
        processors.
        """
        if self.fields is None:
            self.init_sim()
        near2far.load_hdf5(self.fields, fname, '', self.get_filename_prefix())

    def save_near2far(self, fname, near2far):
        """
        Save the Fourier-transformed fields corresponding to the given `near2far` object
        in an HDF5 file of the given `filename` (without the `.h5` suffix). The current
        filename-prefix is prepended automatically.
        """
        if self.fields is None:
            self.init_sim()
        near2far.save_hdf5(self.fields, fname, '', self.get_filename_prefix())

    def load_minus_near2far(self, fname, near2far):
        """
        As `load_near2far`, but negates the Fourier-transformed fields after they are
        loaded. This means that they will be *subtracted* from any future field Fourier
        transforms that are accumulated.
        """
        self.load_near2far(fname, near2far)
        near2far.scale_dfts(-1.0)

    def get_near2far_data(self, near2far):
        """
        Get the Fourier-transformed fields corresponding to the given `near2far` object as
        a `NearToFarData`, which is just a named tuple of NumPy arrays. Note that this
        object is only useful for passing to `load_near2far_data` below and should be
        considered opaque.
        """
        return NearToFarData(F=self.get_dft_data(near2far.F))

    def load_near2far_data(self, near2far, n2fdata):
        """
        Load the Fourier-transformed fields into the `near2far` object (replacing any
        values currently there) from the `NearToFarData` object `n2fdata`. You must load
        from an object that was created by `get_near2far_data` in a simulation of the same
        dimensions (for both the cell and the flux regions) with the same number of
        processors.
        """
        mp._load_dft_data(near2far.F, n2fdata.F)

    def load_minus_near2far_data(self, near2far, n2fdata):
        """
        As `load_near2far_data`, but negates the Fourier-transformed fields after they are
        loaded. This means that they will be *subtracted* from any future field Fourier
        transforms that are accumulated.
        """
        self.load_near2far_data(near2far, n2fdata)
        near2far.scale_dfts(complex(-1.0))

    def add_force(self, *args):
        """
        `add_force(fcen, df, nfreq, freq, ForceRegions...)`  ##sig

        Add a bunch of `ForceRegion`s to the current simulation (initializing the fields
        if they have not yet been initialized), telling Meep to accumulate the appropriate
        field Fourier transforms for `nfreq` equally spaced frequencies covering the
        frequency range `fcen-df/2` to `fcen+df/2` or an array/list `freq` for arbitrarily
        spaced frequencies. Return a `force`object, which you can pass to the functions
        below to get the force spectrum, etcetera.
        """
        args = fix_dft_args(args, 0)
        freq = args[0]
        forces = args[1:]
        force = DftForce(self._add_force, [freq, forces])
        self.dft_objects.append(force)
        return force

    def _add_force(self, freq, forces):
        if self.fields is None:
            self.init_sim()
        return self._add_fluxish_stuff(self.fields.add_dft_force, freq, forces)

    def display_forces(self, *forces):
        """
        Given a number of force objects, this displays a comma-separated table of
        frequencies and force spectra, prefixed by "force1:" or similar (where the number
        is incremented after each run). All of the forces should be for the same
        `fcen`/`df`/`nfreq` or `freq`. The first column are the frequencies, and
        subsequent columns are the force spectra.
        """
        force_freqs = get_force_freqs(forces[0])
        display_csv(self, 'force', zip(force_freqs, *[get_forces(f) for f in forces]))

    def load_force(self, fname, force):
        """
        Load the Fourier-transformed fields into the given force object (replacing any
        values currently there) from an HDF5 file of the given `filename` without the
        `.h5` suffix (the current filename-prefix is prepended automatically). You must
        load from a file that was saved by `save_force` in a simulation of the same
        dimensions for both the cell and the force regions with the same number of
        processors.
        """
        if self.fields is None:
            self.init_sim()
        force.load_hdf5(self.fields, fname, '', self.get_filename_prefix())

    def save_force(self, fname, force):
        """
        Save the Fourier-transformed fields corresponding to the given force object in an
        HDF5 file of the given `filename` without the `.h5` suffix (the current
        filename-prefix is prepended automatically).
        """
        if self.fields is None:
            self.init_sim()
        force.save_hdf5(self.fields, fname, '', self.get_filename_prefix())

    def load_minus_force(self, fname, force):
        """
        As `load_force`, but negates the Fourier-transformed fields after they are loaded.
        This means that they will be *subtracted* from any future field Fourier transforms
        that are accumulated.
        """
        self.load_force(fname, force)
        force.scale_dfts(-1.0)

    def get_force_data(self, force):
        """
        Get the Fourier-transformed fields corresponding to the given force object as a
        `ForceData`, which is just a named tuple of NumPy arrays. Note that this object is
        only useful for passing to `load_force_data` below and should be considered
        opaque.
        """
        return ForceData(offdiag1=self.get_dft_data(force.offdiag1),
                         offdiag2=self.get_dft_data(force.offdiag2),
                         diag=self.get_dft_data(force.diag))

    def load_force_data(self, force, fdata):
        """
        Load the Fourier-transformed fields into the given force object (replacing any
        values currently there) from the `ForceData` object `fdata`. You must load from an
        object that was created by `get_force_data` in a simulation of the same dimensions
        (for both the cell and the flux regions) with the same number of processors.
        """
        mp._load_dft_data(force.offdiag1, fdata.offdiag1)
        mp._load_dft_data(force.offdiag2, fdata.offdiag2)
        mp._load_dft_data(force.diag, fdata.diag)

    def load_minus_force_data(self, force, fdata):
        """
        As `load_force_data`, but negates the Fourier-transformed fields after they are
        loaded. This means that they will be *subtracted* from any future field Fourier
        transforms that are accumulated.
        """
        self.load_force_data(force, fdata)
        force.scale_dfts(complex(-1.0))

    def add_flux(self, *args):
        """
        `add_flux(fcen, df, nfreq, freq, FluxRegions...)` ##sig

        Add a bunch of `FluxRegion`s to the current simulation (initializing the fields if
        they have not yet been initialized), telling Meep to accumulate the appropriate
        field Fourier transforms for `nfreq` equally spaced frequencies covering the
        frequency range `fcen-df/2` to `fcen+df/2` or an array/list `freq` for arbitrarily
        spaced frequencies. Return a *flux object*, which you can pass to the functions
        below to get the flux spectrum, etcetera.
        """
        args = fix_dft_args(args, 0)
        freq = args[0]
        fluxes = args[1:]
        flux = DftFlux(self._add_flux, [freq, fluxes])
        self.dft_objects.append(flux)
        return flux

    def _add_flux(self, freq, fluxes):
        if self.fields is None:
            self.init_sim()
        return self._add_fluxish_stuff(self.fields.add_dft_flux, freq, fluxes)

    def add_mode_monitor(self, *args, **kwargs):
        """
        `add_mode_monitor(fcen, df, nfreq, freq, ModeRegions...)`  ##sig

        Similar to `add_flux`, but for use with `get_eigenmode_coefficients`.
        """
        args = fix_dft_args(args, 0)
        freq = args[0]
        fluxes = args[1:]
        yee_grid = kwargs.get("yee_grid", False)
        flux = DftFlux(self._add_mode_monitor, [freq, fluxes, yee_grid])
        self.dft_objects.append(flux)
        return flux

    def _add_mode_monitor(self, freq, fluxes, yee_grid):
        if self.fields is None:
            self.init_sim()

        if len(fluxes) != 1:
            raise ValueError("add_mode_monitor expected just one ModeRegion. Got {}".format(len(fluxes)))

        region = fluxes[0]
        centered_grid = not yee_grid
        v = mp.Volume(region.center, region.size, dims=self.dimensions, is_cylindrical=self.is_cylindrical)
        d0 = region.direction
        d = self.fields.normal_direction(v.swigobj) if d0 < 0 else d0

        return self.fields.add_mode_monitor(d, v.swigobj, freq, centered_grid)

    def display_fluxes(self, *fluxes):
        """
        Given a number of flux objects, this displays a comma-separated table of
        frequencies and flux spectra, prefixed by "flux1:" or similar (where the number is
        incremented after each run). All of the fluxes should be for the same
        `fcen`/`df`/`nfreq` or `freq`. The first column are the frequencies, and
        subsequent columns are the flux spectra.
        """
        display_csv(self, 'flux', zip(get_flux_freqs(fluxes[0]), *[get_fluxes(f) for f in fluxes]))

    def load_flux(self, fname, flux):
        """
        Load the Fourier-transformed fields into the given flux object (replacing any
        values currently there) from an HDF5 file of the given `filename` without the
        `.h5` suffix (the current filename-prefix is prepended automatically). You must
        load from a file that was saved by `save_flux` in a simulation of the same
        dimensions (for both the cell and the flux regions) with the same number of
        processors.
        """
        if self.fields is None:
            self.init_sim()

        flux.load_hdf5(self.fields, fname, '', self.get_filename_prefix())

    load_mode = load_flux

    def save_flux(self, fname, flux):
        """
        Save the Fourier-transformed fields corresponding to the given flux object in an
        HDF5 file of the given `filename` without the `.h5` suffix (the current
        filename-prefix is prepended automatically).
        """
        if self.fields is None:
            self.init_sim()

        flux.save_hdf5(self.fields, fname, '', self.get_filename_prefix())

    save_mode = save_flux

    def load_minus_flux(self, fname, flux):
        """
        As `load_flux`, but negates the Fourier-transformed fields after they are loaded.
        This means that they will be *subtracted* from any future field Fourier transforms
        that are accumulated.
        """
        self.load_flux(fname, flux)
        flux.scale_dfts(complex(-1.0))

    load_minus_mode = load_minus_flux

    def get_flux_data(self, flux):
        """
        Get the Fourier-transformed fields corresponding to the given flux object as a
        `FluxData`, which is just a named tuple of NumPy arrays. Note that this object is
        only useful for passing to `load_flux_data` below and should be considered opaque.
        """
        return FluxData(E=self.get_dft_data(flux.E), H=self.get_dft_data(flux.H))

    get_mode_data = get_flux_data

    def load_flux_data(self, flux, fdata):
        """
        Load the Fourier-transformed fields into the given flux object (replacing any
        values currently there) from the `FluxData` object `fdata`. You must load from an
        object that was created by `get_flux_data` in a simulation of the same dimensions
        (for both the cell and the flux regions) with the same number of processors.
        """
        mp._load_dft_data(flux.E, fdata.E)
        mp._load_dft_data(flux.H, fdata.H)

    load_mode_data = load_flux_data

    def load_minus_flux_data(self, flux, fdata):
        """
        As `load_flux_data`, but negates the Fourier-transformed fields after they are
        loaded. This means that they will be *subtracted* from any future field Fourier
        transforms that are accumulated.
        """
        self.load_flux_data(flux, fdata)
        flux.scale_dfts(complex(-1.0))

    load_minus_mode_data = load_minus_flux_data

    def flux_in_box(self, d, box=None, center=None, size=None):
        """
        Given a `direction` constant, and a `mp.Volume`, returns the flux (the integral of
        $\\Re [\\mathbf{E}^* \\times \\mathbf{H}]$) in that volume. Most commonly, you specify
        a volume that is a plane or a line, and a direction perpendicular to it, e.g.

        `flux_in_box(d=mp.X,mp.Volume(center=mp.Vector3(0,0,0),size=mp.Vector3(0,1,1)))`

        If the `center` and `size` arguments are provided instead of `box`, Meep will
        construct the appropriate volume for you.
        """
        if self.fields is None:
            raise RuntimeError('Fields must be initialized before using flux_in_box')

        box = self._volume_from_kwargs(box, center, size)

        return self.fields.flux_in_box(d, box)

    def electric_energy_in_box(self, box=None, center=None, size=None):
        """
        Given a `mp.Volume`, returns the integral of the electric-field energy
        $\\mathbf{E}^* \\cdot \\mathbf{D}/2$ in the given volume. If the volume has zero size
        along a dimension, a lower-dimensional integral is used. If the `center` and
        `size` arguments are provided instead of `box`, Meep will construct the
        appropriate volume for you. Note: in cylindrical coordinates $(r,\\phi,z)$, the
        integrand is
        [multiplied](https://en.wikipedia.org/wiki/Cylindrical_coordinate_system#Line_and_volume_elements)
        by the circumference $2\\pi r$, or equivalently the integral is over an annular
        volume.
        """
        if self.fields is None:
            raise RuntimeError('Fields must be initialized before using electric_energy_in_box')

        box = self._volume_from_kwargs(box, center, size)

        return self.fields.electric_energy_in_box(box)

    def magnetic_energy_in_box(self, box=None, center=None, size=None):
        """
        Given a `mp.Volume`, returns the integral of the magnetic-field energy
        $\\mathbf{H}^* \\cdot \\mathbf{B}/2$ in the given volume. If the volume has zero size
        along a dimension, a lower-dimensional integral is used. If the `center` and
        `size` arguments are provided instead of `box`, Meep will construct the
        appropriate volume for you. Note: in cylindrical coordinates $(r,\\phi,z)$, the
        integrand is
        [multiplied](https://en.wikipedia.org/wiki/Cylindrical_coordinate_system#Line_and_volume_elements)
        by the circumference $2\\pi r$, or equivalently the integral is over an annular
        volume.
        """
        if self.fields is None:
            raise RuntimeError('Fields must be initialized before using magnetic_energy_in_box')

        box = self._volume_from_kwargs(box, center, size)

        return self.fields.magnetic_energy_in_box(box)

    def field_energy_in_box(self, box=None, center=None, size=None):
        """
        Given a `mp.Volume`, returns the integral of the electric- and magnetic-field
        energy $\\mathbf{E}^* \\cdot \\mathbf{D}/2 + \\mathbf{H}^* \\cdot \\mathbf{B}/2$ in the
        given volume. If the volume has zero size along a dimension, a lower-dimensional
        integral is used. If the `center` and `size` arguments are provided instead of
        `box`, Meep will construct the appropriate volume for you. Note: in cylindrical
        coordinates $(r,\\phi,z)$, the integrand is
        [multiplied](https://en.wikipedia.org/wiki/Cylindrical_coordinate_system#Line_and_volume_elements)
        by the circumference $2\\pi r$, or equivalently the integral is over an annular
        volume.
        """
        if self.fields is None:
            raise RuntimeError('Fields must be initialized before using field_energy_in_box')

        box = self._volume_from_kwargs(box, center, size)

        return self.fields.field_energy_in_box(box)

    def modal_volume_in_box(self, box=None, center=None, size=None):
        """
        Given a `mp.Volume`, returns the instantaneous modal volume
        according to the Purcell-effect definition:
        $\\left(\\int\\varepsilon|\\mathbf{E}|^2\\right)/\\left(\\max{\\varepsilon|\\mathbf{E}|^2}\\right)$.
        If no volume argument is provided, the entire cell is used by
        default. If the `center` and `size` arguments are provided
        instead of `box`, Meep will construct the appropriate volume
        for you.

        Note that if you are at a fixed frequency and you use complex fields (via
        Bloch-periodic boundary conditions or `fields_complex=True`), then one half of the
        flux or energy integrals above corresponds to the time average of the flux or
        energy for a simulation with real fields.

        Often, you want the integration box to be the entire cell. A useful function to
        return this box, which you can then use for the `box` arguments above, is
        `Simulation.total_volume()`.

        One versatile feature is that you can supply an arbitrary function
        $f(\\mathbf{x},c_1,c_2,\\ldots)$ of position $\\mathbf{x}$ and various field
        components $c_1,\\ldots$ and ask Meep to integrate it over a given volume, find its
        maximum, or output it (via `output_field_function`, described later). This is done
        via the functions:
        """
        if self.fields is None:
            raise RuntimeError('Fields must be initialized before using modal_volume_in_box')

        try:
            box = self._volume_from_kwargs(box, center, size)
        except ValueError:
            box = self.fields.total_volume()

        return self.fields.modal_volume_in_box(box)

    def solve_cw(self, tol=1e-8, maxiters=10000, L=2):
        if self.fields is None:
            raise RuntimeError('Fields must be initialized before using solve_cw')
        self._evaluate_dft_objects()
        return self.fields.solve_cw(tol, maxiters, L)

    def solve_eigfreq(self, tol=1e-7, maxiters=100, guessfreq=None, cwtol=None, cwmaxiters=10000, L=10):
        if self.fields is None:
            raise RuntimeError('Fields must be initialized before using solve_cw')
        if cwtol is None:
            cwtol = tol * 1e-3 # solve CW problems much more accurately than eigenvalue tolerance
        self._evaluate_dft_objects()
        eigfreq = np.array(0, dtype=np.complex128)
        if guessfreq is None:
            self.fields.solve_cw(cwtol, cwmaxiters, L, eigfreq, tol, maxiters)
        else:
            self.fields.solve_cw(cwtol, cwmaxiters, guessfreq, L, eigfreq, tol, maxiters)
        return eigfreq.item()

    def _add_fluxish_stuff(self, add_dft_stuff, freq, stufflist, *args):
        vol_list = None

        for s in stufflist:
            v = Volume(center=s.center, size=s.size, dims=self.dimensions,
                       is_cylindrical=self.is_cylindrical)
            d0 = s.direction
            d = self.fields.normal_direction(v.swigobj) if d0 < 0 else d0
            c = mp.direction_component(mp.Sx, d)
            v2 = Volume(center=s.center, size=s.size, dims=self.dimensions,
                        is_cylindrical=self.is_cylindrical).swigobj
            vol_list = mp.make_volume_list(v2, c, s.weight, vol_list)

        stuff = add_dft_stuff(vol_list, freq, *args)
        vol_list.__swig_destroy__(vol_list)

        return stuff

    def output_component(self, c, h5file=None, frequency=0):
        if self.fields is None:
            raise RuntimeError("Fields must be initialized before calling output_component")

        vol = self.fields.total_volume() if self.output_volume is None else self.output_volume
        h5 = self.output_append_h5 if h5file is None else h5file
        append = h5file is None and self.output_append_h5 is not None

        self.fields.output_hdf5(c, vol, h5, append, self.output_single_precision,self.get_filename_prefix(), frequency)

        if h5file is None:
            nm = self.fields.h5file_name(mp.component_name(c), self.get_filename_prefix(), True)
            if c == mp.Dielectric:
                self.last_eps_filename = nm
            self.output_h5_hook(nm)

    def output_components(self, fname, *components):
        if self.fields is None:
            raise RuntimeError("Fields must be initialized before calling output_component")

        if self.output_append_h5 is None:
            f = self.fields.open_h5file(fname, mp.h5file.WRITE, self.get_filename_prefix(), True)
        else:
            f = None

        for c in components:
            self.output_component(c, h5file=f)
            if self.output_append_h5 is None:
                f.prevent_deadlock()

        if self.output_append_h5 is None:
            self.output_h5_hook(self.fields.h5file_name(fname, self.get_filename_prefix(), True))

    def h5topng(self, rm_h5, option, *step_funcs):
        opts = "h5topng {}".format(option)
        cmd = re.sub(r'\$EPS', self.last_eps_filename, opts)
        return convert_h5(rm_h5, cmd, *step_funcs)

    def get_array(self, component=None, vol=None, center=None, size=None, cmplx=None, arr=None, frequency=0, snap=False):
        """
        Takes as input a subregion of the cell and the field/material component. The
        method returns a NumPy array containing values of the field/material at the
        current simulation time.

        **Parameters:**

        + `vol`: `Volume`; the orthogonal subregion/slice of the computational volume. The
          return value of `get_array` has the same dimensions as the `Volume`'s `size`
          attribute. If `None` (default), then a `size` and `center` must be specified.

        + `center`, `size` : `Vector3`; if both are specified, the library will construct
          an appropriate `Volume`. This is a convenience feature and alternative to
          supplying a `Volume`.

        + `component`: field/material component (i.e., `mp.Ex`, `mp.Hy`, `mp.Sz`,
          `mp.Dielectric`, etc). Defaults to `None`.

        + `cmplx`: `boolean`; if `True`, return complex-valued data otherwise return
          real-valued data (default).

        + `arr`: optional field to pass a pre-allocated NumPy array of the correct size,
          which will be overwritten with the field/material data instead of allocating a
          new array.  Normally, this will be the array returned from a previous call to
          `get_array` for a similar slice, allowing one to re-use `arr` (e.g., when
          fetching the same slice repeatedly at different times).

        + `frequency`: optional frequency point over which the average eigenvalue of the
          $\\varepsilon$ and $\\mu$ tensors are evaluated (defaults to 0).

        + `snap`: By default, the elements of the grid slice are obtained using a bilinear
          interpolation of the nearest Yee grid points. Empty dimensions of the grid slice
          are "collapsed" into a single element. However, if `snap` is set to `True`, this
          interpolation behavior is disabled and the grid slice is instead "snapped"
          everywhere to the nearest grid point. (Empty slice dimensions are still of size
          one.) This feature is mainly useful for comparing results with the
          [`output_` routines](#output-functions) (e.g., `output_epsilon`, `output_efield_z`, etc.).

        For convenience, the following wrappers for `get_array` over the entire cell are
        available: `get_epsilon()`, `get_mu()`, `get_hpwr()`, `get_dpwr()`,
        `get_tot_pwr()`, `get_Xfield()`, `get_Xfield_x()`, `get_Xfield_y()`,
        `get_Xfield_z()`, `get_Xfield_r()`, `get_Xfield_p()` where `X` is one of `h`, `b`,
        `e`, `d`, or `s`. The routines `get_Xfield_*` all return an array type consistent
        with the fields (real or complex). The routines `get_epsilon()` and `get_mu()`
        accept the optional argument `frequency` (defaults to 0) and all routines accept
        `snap` (defaults to `False`).

        **Note on array-slice dimensions:** The routines `get_epsilon`, `get_Xfield_z`,
        etc. use as default `size=meep.Simulation.fields.total_volume()` which for
        simulations involving Bloch-periodic boundaries (via `k_point`) will result in
        arrays that have slightly *different* dimensions than e.g.
        `get_array(center=meep.Vector3(), size=cell_size, component=meep.Dielectric`, etc.
        (i.e., the slice spans the entire cell volume `cell_size`). Neither of these
        approaches is "wrong", they are just slightly different methods of fetching the
        boundaries. The key point is that if you pass the same value for the `size`
        parameter, or use the default, the slicing routines always give you the same-size
        array for all components. You should *not* try to predict the exact size of these
        arrays; rather, you should simply rely on Meep's output.
        """
        if component is None:
            raise ValueError("component is required")
        if isinstance(component, mp.Volume) or isinstance(component, mp.volume):
            raise ValueError("The first argument must be the component")

        dim_sizes = np.zeros(3, dtype=np.uintp)

        if vol is None and center is None and size is None:
            v = self.fields.total_volume()
        else:
            v = self._volume_from_kwargs(vol, center, size)

        _, dirs = mp._get_array_slice_dimensions(self.fields, v, dim_sizes, not snap, snap)

        dims = [s for s in dim_sizes if s != 0]

        if cmplx is None:
            cmplx = frequency != 0 or (component < mp.Dielectric and not self.fields.is_real)

        if arr is not None:
            if cmplx and not np.iscomplexobj(arr):
                raise ValueError("Requested a complex slice, but provided array of type {}.".format(arr.dtype))

            for a, b in zip(arr.shape, dims):
                if a != b:
                    fmt = "Expected dimensions {}, but got {}"
                    raise ValueError(fmt.format(dims, arr.shape))

            arr = np.require(arr, requirements=['C', 'W'])

        else:
            arr = np.zeros(dims, dtype=np.complex128 if cmplx else np.float64)

        if np.iscomplexobj(arr):
            self.fields.get_complex_array_slice(v, component, arr, frequency, snap)
        else:
            self.fields.get_array_slice(v, component, arr, frequency, snap)

        return arr

    def get_dft_array(self, dft_obj, component, num_freq):
        """
        Returns the Fourier-transformed fields as a NumPy array.

        **Parameters:**

        + `dft_obj`: a `dft_flux`, `dft_force`, `dft_fields`, or `dft_near2far` object
          obtained from calling the appropriate `add` function (e.g., `mp.add_flux`).

        + `component`: a field component (e.g., `mp.Ez`).

        + `num_freq`: the index of the frequency. An integer in the range `0...nfreq-1`,
          where `nfreq` is the number of frequencies stored in `dft_obj` as set by the
          `nfreq` parameter to `add_dft_fields`, `add_flux`, etc.
        """
        if not self.dft_objects:
            raise RuntimeError('DFT monitor dft_obj must be initialized before calling get_dft_array')

        if hasattr(dft_obj, 'swigobj'):
            dft_swigobj = dft_obj.swigobj
        else:
            dft_swigobj = dft_obj

        if type(dft_swigobj) is mp.dft_fields:
            return mp.get_dft_fields_array(self.fields, dft_swigobj, component, num_freq)
        elif type(dft_swigobj) is mp.dft_flux:
            return mp.get_dft_flux_array(self.fields, dft_swigobj, component, num_freq)
        elif type(dft_swigobj) is mp.dft_force:
            return mp.get_dft_force_array(self.fields, dft_swigobj, component, num_freq)
        elif type(dft_swigobj) is mp.dft_near2far:
            return mp.get_dft_near2far_array(self.fields, dft_swigobj, component, num_freq)
        else:
            raise ValueError("Invalid type of dft object: {}".format(dft_swigobj))

    def get_source(self, component, vol=None, center=None, size=None):
        """
        Return an array of complex values of the [source](#source) amplitude for
        `component` over the given `vol` or `center`/`size`. The array has the same
        dimensions as that returned by [`get_array`](#array-slices).
        Not supported for [cylindrical coordinates](Python_Tutorials/Cylindrical_Coordinates.md).
        """
        if vol is None and center is None and size is None:
            v = self.fields.total_volume()
        else:
            v = self._volume_from_kwargs(vol, center, size)
        dim_sizes = np.zeros(3, dtype=np.uintp)
        mp._get_array_slice_dimensions(self.fields, v, dim_sizes, True, False)
        dims = [s for s in dim_sizes if s != 0]
        arr = np.zeros(dims, dtype=np.complex128)
        self.fields.get_source_slice(v, component, arr)
        return arr

    def get_array_metadata(self, vol=None, center=None, size=None, dft_cell=None,
                           return_pw=False):
        """
        This routine provides geometric information useful for interpreting the arrays
        returned by `get_array` or `get_dft_array` for the spatial region defined by `vol`
        or `center`/`size`. In both cases, the return value is a tuple `(x,y,z,w)`, where:

        + `x,y,z` are 1d NumPy arrays storing the $x,y,z$ coordinates of the points in the
          grid slice
        + `w` is a NumPy array of the same dimensions as the array returned by
          `get_array`/`get_dft_array`, whose entries are the weights in a cubature rule
          for integrating over the spatial region (with the points in the cubature rule
          being just the grid points contained in the region). Thus, if $Q(\\mathbf{x})$ is
          some spatially-varying quantity whose value at the $n$th grid point is $Q_n$,
          the integral of $Q$ over the region may be approximated by the sum:

        $$ \\int_{\\mathcal V} Q(\\mathbf{x})d\\mathbf{x} \\approx \\sum_{n} w_n Q_n.$$

        This is a 1-, 2-, or 3-dimensional integral depending on the number of dimensions
        in which $\\mathcal{V}$ has zero extent. If the $Q_n$ samples are stored in an
        array `Q` of the same dimensions as `w`, then evaluating the sum on the RHS is
        just one line: `np.sum(w*Q).`

        A convenience parameter `dft_cell` is provided as an alternative to `vol` or
        `center`/`size`. Set `dft_cell` to a `dft_flux` or `dft_fields` object to define the
        region covered by the array. If the `dft_cell` argument is provided then all other
        arguments related to the spatial region (`vol`, `center`, and `size`) are ignored.
        If no arguments are provided, then the entire cell is used.

        For empty dimensions of the grid slice `get_array_metadata` will collapse
        the *two* elements corresponding to the nearest Yee grid points into a *single*
        element using linear interpolation.

        If `return_pw=True`, the return value is a 2-tuple `(p,w)` where `p` (points) is a
        list of `mp.Vector3`s with the same dimensions as `w` (weights). Otherwise, by
        default the return value is a 4-tuple `(x,y,z,w)`.
        """
        if dft_cell:
            vol = dft_cell.where
        if vol is None and center is None and size is None:
            v = self.fields.total_volume()
        else:
            v = self._volume_from_kwargs(vol, center, size)
        xyzw_vector = self.fields.get_array_metadata(v)
        offset, tics = 0, []
        for n in range(3):
            N = int(xyzw_vector[offset])
            tics.append( xyzw_vector[offset+1:offset+1+N] )
            offset += 1+N
        wshape = [len(t) for t in tics if len(t)>1]
        weights = np.reshape(xyzw_vector[offset:], wshape)
        if return_pw:
            points=[ mp.Vector3(x,y,z) for x in tics[0] for y in tics[1] for z in tics[2] ]
            return points,weights
        return tuple(tics) + (weights,)

    def get_array_slice_dimensions(self, component, vol=None, center=None, size=None):
        """
        Computes the dimensions of an array slice for a particular `component` (`mp.Ez`, `mp.Ey`, etc.).

        Accepts either a volume object (`vol`), or a `center` and `size` `Vector3` pair.

        Returns a tuple containing the dimensions (`dim_sizes`), a `Vector3` object
        corresponding to the minimum corner of the volume (`min_corner`),
        and a `Vector3` object corresponding to the maximum corner (`max_corner`).
        """
        if vol is None and center is None and size is None:
            v = self.fields.total_volume()
        else:
            v = self._volume_from_kwargs(vol, center, size)
        dim_sizes = np.zeros(3, dtype=np.uintp)
        corners = []
        _,_ = mp._get_array_slice_dimensions(self.fields, v, dim_sizes, False, False, component, corners)
        dim_sizes[dim_sizes==0] = 1
        min_corner = corners[0]
        max_corner = corners[1]
        return dim_sizes, min_corner, max_corner

    def get_eigenmode_coefficients(self, flux, bands, eig_parity=mp.NO_PARITY, eig_vol=None,
                                   eig_resolution=0, eig_tolerance=1e-12, kpoint_func=None, direction=mp.AUTOMATIC):
        """
        Given a flux object and list of band indices `bands` or `DiffractedPlanewave`, return a `namedtuple` with the
        following fields:

        + `alpha`: the complex eigenmode coefficients as a 3d NumPy array of size
          (`len(bands)`, `flux.nfreqs`, `2`). The last/third dimension refers to modes
          propagating in the forward (+) or backward (-) directions.
        + `vgrp`: the group velocity as a NumPy array.
        + `kpoints`: a list of `mp.Vector3`s of the `kpoint` used in the mode calculation.
        + `kdom`: a list of `mp.Vector3`s of the mode's dominant wavevector.
        + `cscale`: a NumPy array of each mode's scaling coefficient. Useful for adjoint
          calculations.
        """
        if self.fields is None:
            raise ValueError("Fields must be initialized before calling get_eigenmode_coefficients")
        if eig_vol is None:
            eig_vol = flux.where
        else:
            eig_vol = self._volume_from_kwargs(vol=eig_vol)
        if direction is None or direction == mp.AUTOMATIC:
            direction = flux.normal_direction

        try:
            bands_list_range = isinstance(bands, (list,range))
        except TypeError:
            bands_list_range = isinstance(bands, list)

        if bands_list_range:
            num_bands = len(bands)
            coeffs = np.zeros(2 * num_bands * flux.freq.size(), dtype=np.complex128)
            vgrp = np.zeros(num_bands * flux.freq.size())
            cscale = np.zeros(num_bands * flux.freq.size())

            kpoints, kdom = mp.get_eigenmode_coefficients_and_kpoints(
                self.fields,
                flux.swigobj,
                eig_vol,
                np.array(bands, dtype=np.intc),
                eig_parity,
                eig_resolution,
                eig_tolerance,
                coeffs,
                vgrp,
                kpoint_func,
                cscale,
                direction
            )
        elif isinstance(bands, DiffractedPlanewave):
            num_bands = 1
            coeffs = np.zeros(2 * num_bands * flux.freq.size(), dtype=np.complex128)
            vgrp = np.zeros(num_bands * flux.freq.size())
            cscale = np.zeros(num_bands * flux.freq.size())
            diffractedplanewave = bands_to_diffractedplanewave(flux.where, bands)

            kpoints, kdom = mp.get_eigenmode_coefficients_and_kpoints(
                self.fields,
                flux.swigobj,
                eig_vol,
                diffractedplanewave,
                eig_parity,
                eig_resolution,
                eig_tolerance,
                coeffs,
                vgrp,
                kpoint_func,
                cscale,
                direction
            )
        else:
            raise TypeError("get_eigenmode_coefficients: bands must be either a list or DiffractedPlanewave object")

        return EigCoeffsResult(np.reshape(coeffs, (num_bands, flux.freq.size(), 2)), vgrp, kpoints, kdom, cscale)

    def get_eigenmode(self, frequency, direction, where, band_num, kpoint, eig_vol=None, match_frequency=True,
                      parity=mp.NO_PARITY, resolution=0, eigensolver_tol=1e-12):
        """
        The parameters of this routine are the same as that of
        `get_eigenmode_coefficients` or `EigenModeSource`, but this function returns an
        object that can be used to inspect the computed mode.  In particular, it returns
        an `EigenmodeData` instance with the following fields:

        + `band_num`: same as a single element of the `bands` parameter
        + `freq`: the computed frequency, same as the `frequency` input parameter if
          `match_frequency=True`
        + `group_velocity`: the group velocity of the mode in `direction`
        + `k`: the Bloch wavevector of the mode in `direction`
        + `kdom`: the dominant planewave of mode `band_num`
        + `amplitude(point, component)`: the (complex) value of the given E or H field
          `component` (`Ex`, `Hy`, etcetera) at a particular `point` (a `Vector3`) in
          space (interpreted with Bloch-periodic boundary conditions if you give a point
          outside the original `eig_vol`).

        If `match_frequency=False` or `kpoint` is not zero in the given `direction`, the
        `frequency` input parameter is ignored.
        """

        if self.fields is None:
            raise ValueError("Fields must be initialized before calling get_eigenmode")

        where = self._volume_from_kwargs(vol=where)
        if eig_vol is None:
            eig_vol = where
        else:
            eig_vol = self._volume_from_kwargs(vol=eig_vol)

        swig_kpoint = mp.vec(kpoint.x, kpoint.y, kpoint.z)
        kdom = np.zeros(3)
        emdata = mp._get_eigenmode(self.fields, frequency, direction, where, eig_vol, band_num, swig_kpoint,
                                   match_frequency, parity, resolution, eigensolver_tol, kdom)
        Gk = mp._get_eigenmode_Gk(emdata)

        return EigenmodeData(emdata.band_num, emdata.frequency, emdata.group_velocity, Gk,
                             emdata, mp.Vector3(kdom[0], kdom[1], kdom[2]))

    def output_field_function(self, name, cs, func, real_only=False, h5file=None):
        """
        Output the field function `func` to an HDF5 file in the datasets named `name*.r`
        and `name*.i` for the real and imaginary parts. Similar to
        `integrate_field_function`, `func` is a function of position (a `Vector3`) and the
        field components corresponding to `cs`: a list of `component` constants. If
        `real_only` is True, only outputs the real part of `func`.
        """
        if self.fields is None:
            raise RuntimeError("Fields must be initialized before calling output_field_function")

        ov = self.output_volume if self.output_volume else self.fields.total_volume()
        h5 = self.output_append_h5 if h5file is None else h5file
        append = h5file is None and self.output_append_h5 is not None

        self.fields.output_hdf5(name, [cs, func], ov, h5, append, self.output_single_precision,
                                self.get_filename_prefix(), real_only)
        if h5file is None:
            self.output_h5_hook(self.fields.h5file_name(name, self.get_filename_prefix(), True))

    def _get_field_function_volume(self, where=None, center=None, size=None):
        try:
            where = self._volume_from_kwargs(where, center, size)
        except ValueError:
            where = self.fields.total_volume()

        return where

    def integrate_field_function(self, cs, func, where=None, center=None, size=None):
        """
        Returns the integral of the complex-valued function `func` over the `Volume`
        specified by `where` (defaults to entire cell) for the `meep::fields` contained in
        the `Simulation` instance that calls this method. `func` is a function of position
        (a `Vector3`, its first argument) and zero or more field components specified by
        `cs`: a list of `component` constants. `func` can be real- or complex-valued. The
        volume can optionally be specified via the `center` and `size` arguments.

        If any dimension of `where` is zero, that dimension is not integrated over. In
        this way you can specify 1d, 2d, or 3d integrals.

        Note: in cylindrical coordinates $(r,\\phi,z)$, the integrand is
        [multiplied](https://en.wikipedia.org/wiki/Cylindrical_coordinate_system#Line_and_volume_elements)
        by the circumference $2\\pi r$, or equivalently the integral is over an annular
        volume.
        """
        where = self._get_field_function_volume(where, center, size)
        return self.fields.integrate([cs, func], where)

    def integrate2_field_function(self, fields2, cs1, cs2, func, where=None, center=None, size=None):
        """
        Similar to `integrate_field_function`, but takes additional parameters `fields2`
        and `cs2`. `fields2` is a `meep::fields*` object similar to the global `fields`
        variable (see below) specifying the fields from another simulation. `cs1` is a
        list of components to integrate with from the `meep::fields` instance in
        `Simulation.fields`, as for `integrate_field_function`, while `cs2` is a list of
        components to integrate from `fields2`. Similar to `integrate_field_function`,
        `func` is a function that returns an number given arguments consisting of: the
        position vector, followed by the values of the components specified by `cs1` (in
        order), followed by the values of the components specified by `cs2` (in order).
        The volume can optionally be specified via the `center` and `size` arguments.

        To get two fields in memory at once for `integrate2_field_function`, the easiest
        way is to run one simulation within a given Python file, then save the results in
        another fields variable, then run a second simulation. This would look something
        like:

        ```py
        ...set up and run first simulation...
        fields2 = sim.fields # save the fields in a variable
        sim.fields = None    # prevent the fields from getting deallocated by reset-meep
        sim.reset_meep()
        ...set up and run second simulation...
        ```

        It is also possible to timestep both fields simultaneously (e.g. doing one
        timestep of one simulation then one timestep of another simulation, and so on, but
        this requires you to call much lower-level functions like `fields_step()`.
        """
        where = self._get_field_function_volume(where, center, size)
        return self.fields.integrate2(fields2, [cs1, cs2, func], where)

    def max_abs_field_function(self, cs, func, where=None, center=None, size=None):
        """
        As `integrate_field_function`, but returns the maximum absolute value of `func` in
        the volume `where` instead of its integral.

        The integration is performed by summing over the grid points with a simple
        trapezoidal rule, and the maximum is similarly over the grid points. See [Field
        Functions](Field_Functions.md) for examples of how to call
        `integrate_field_function` and `max_abs_field_function`. See [Synchronizing the
        Magnetic and Electric Fields](Synchronizing_the_Magnetic_and_Electric_Fields.md)
        if you want to do computations combining the electric and magnetic fields. The
        volume can optionally be specified via the `center` and `size` arguments.

        Occasionally, one wants to compute an integral that combines fields from two
        separate simulations (e.g. for nonlinear coupled-mode calculations). This
        functionality is supported in Meep, as long as the two simulations have the *same*
        cell, the same resolution, the same boundary conditions and symmetries (if any),
        and the same PML layers (if any).
        """
        where = self._get_field_function_volume(where, center, size)
        return self.fields.max_abs([cs, func], where)

    def change_k_point(self, k):
        """
        Change the `k_point` (the Bloch periodicity).
        """
        self.k_point = k

        if self.fields:
            needs_complex_fields = not (not self.k_point or self.k_point == mp.Vector3())

            if needs_complex_fields and self.fields.is_real:
                self.fields = None
                self._is_initialized = False
                self.init_sim()
            else:
                if self.k_point:
                    self.fields.use_bloch(py_v3_to_vec(self.dimensions, self.k_point, self.is_cylindrical))

    def change_sources(self, new_sources):
        """
        Change the list of sources in `Simulation.sources` to `new_sources`, and changes
        the sources used for the current simulation. `new_sources` must be a list of
        `Source` objects.
        """
        self.sources = new_sources
        if self.fields:
            self.fields.remove_sources()
            self.add_sources()

    def reset_meep(self):
        """
        Reset all of Meep's parameters, deleting the fields, structures, etcetera, from
        memory as if you had not run any computations.
        """
        self.fields = None
        self.structure = None
        self.dft_objects = []
        self._is_initialized = False

    def restart_fields(self):
        """
        Restart the fields at time zero, with zero fields. Does *not* reset the Fourier
        transforms of the flux planes, which continue to be accumulated.
        """
        if self.fields is not None:
            self.fields.t = 0
            self.fields.zero_fields()
        else:
            self._is_initialized = False
            self.init_sim()

    def run(self, *step_funcs, **kwargs):
        """
        `run(step_functions..., until=condition/time)`  ##sig-keep

        Run the simulation until a certain time or condition, calling the given step
        functions (if any) at each timestep. The keyword argument `until` is *either* a
        number, in which case it is an additional time (in Meep units) to run for, *or* it
        is a function (of no arguments) which returns `True` when the simulation should
        stop. `until` can also be a list of stopping conditions which may include a number
        of additional functions.

        `run(step_functions..., until_after_sources=condition/time)`  ##sig-keep

        Run the simulation until all sources have turned off, calling the given step
        functions (if any) at each timestep. The keyword argument `until_after_sources` is
        either a number, in which case it is an *additional* time (in Meep units) to run
        for after the sources are off, *or* it is a function (of no arguments). In the
        latter case, the simulation runs until the sources are off *and* `condition`
        returns `True`. Like `until` above, `until_after_sources` can take a list of
        stopping conditions.
        """
        until = kwargs.pop('until', None)
        until_after_sources = kwargs.pop('until_after_sources', None)

        if self.fields is None:
            self.init_sim()

        self._evaluate_dft_objects()
        self._check_material_frequencies()

        if kwargs:
            raise ValueError("Unrecognized keyword arguments: {}".format(kwargs.keys()))

        if until_after_sources is not None:
            self._run_sources_until(until_after_sources, step_funcs)
        elif until is not None:
            self._run_until(until, step_funcs)
        else:
            raise ValueError("Invalid run configuration")

    def print_times(self):
        """
        Call after running a simulation to print the times spent on various types of work.
        Example output:

        ```
        Field time usage:
                connecting chunks: 0.0156826 s +/- 0.002525 s
                    time stepping: 0.996411 s +/- 0.232147 s
               copying boundaries: 0.148588 s +/- 0.0390397 s
            all-all communication: 1.39423 s +/- 0.581098 s
                1-1 communication: 0.136174 s +/- 0.0107685 s
             Fourier transforming: 0.0321625 s +/- 0.0614168 s
                  MPB mode solver: 0.348019 s +/- 0.370068 s
                  everything else: 0.207387 s +/- 0.0164821 s
        ```
        """
        if self.fields:
            self.fields.print_times()

    def mean_time_spent_on(self, time_sink):
        """
        Return the mean time spent by all processes for a type of work `time_sink` which
        can be one of the following integer constants:
        * meep.Stepping ("time stepping")
        * meep.Connecting ("connecting chunks")
        * meep.Boundaries ("copying boundaries")
        * meep.MpiAllTime ("all-all communication")
        * meep.MpiOneTime ("1-1 communication")
        * meep.FieldOutput ("outputting fields")
        * meep.FourierTransforming ("Fourier transforming")
        * meep.MPBTime ("MPB mode solver")
        * meep.GetFarfieldsTime ("far-field transform")
        * meep.FieldUpdateB ("updating B field")
        * meep.FieldUpdateH ("updating H field")
        * meep.FieldUpdateD ("updating D field")
        * meep.FieldUpdateE ("updating E field")
        * meep.BoundarySteppingB ("boundary stepping B")
        * meep.BoundarySteppingWH ("boundary stepping WH")
        * meep.BoundarySteppingPH ("boundary stepping PH")
        * meep.BoundarySteppingH ("boundary stepping H")
        * meep.BoundarySteppingD ("boundary stepping D")
        * meep.BoundarySteppingWE ("boundary stepping WE")
        * meep.BoundarySteppingPE ("boundary stepping PE")
        * meep.BoundarySteppingE ("boundary stepping E")
        * meep.Other ("everything else")
        """
        return self.fields.mean_time_spent_on(time_sink)

    def time_spent_on(self, time_sink):
        """
        Return a list of times spent by each process for a type of work `time_sink` which
        can be one of ten integer values `0`-`9`: (`0`) connecting chunks, (`1`) time stepping,
        (`2`) copying boundaries, (`3`) MPI all-to-all communication/synchronization,
        (`4`) MPI one-to-one communication, (`5`) field output, (`6`) Fourier transforming,
        (`7`) MPB mode solver, (`8`) near-to-far field transformation, and (`9`) other.
        """
        return self.fields.time_spent_on(time_sink)

    def get_timing_data(self):
        """
        Returns a dictionary that maps each `time_sink` to a list with one entry
        per process. The entries in the list correspond to the total amount of
        time in seconds spent on a particular type of operation. The set of
        valid time sinks is the same as for `mean_time_spent_on`.
        """
        return self.fields.get_timing_data()

    def output_times(self, fname):
        """
        Call after running a simulation to output to a file with filename `fname` the
        times spent on various types of work as CSV (comma separated values) with headers
        for each column and one row per process.
        """
        if self.fields:
            if not fname.endswith('.csv'):
                fname += '.csv'
            self.fields.output_times(fname)

    def get_epsilon(self,frequency=0,snap=False):
        return self.get_array(component=mp.Dielectric,frequency=frequency,snap=snap)

    def get_mu(self,frequency=0,snap=False):
        return self.get_array(component=mp.Permeability,frequency=frequency,snap=snap)

    def get_hpwr(self,snap=False):
        return self.get_array(component=mp.H_EnergyDensity,snap=snap)

    def get_dpwr(self,snap=False):
        return self.get_array(component=mp.D_EnergyDensity,snap=snap)

    def get_tot_pwr(self,snap=False):
        return self.get_array(component=mp.EnergyDensity,snap=snap)

    def get_hfield(self,snap=False):
        if self.is_cylindrical:
            r = self.get_array(mp.Hr, cmplx=not self.fields.is_real, snap=snap)
            p = self.get_array(mp.Hp, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([r, p], axis=-1)
        else:
            x = self.get_array(mp.Hx, cmplx=not self.fields.is_real, snap=snap)
            y = self.get_array(mp.Hy, cmplx=not self.fields.is_real, snap=snap)
            z = self.get_array(mp.Hz, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([x, y, z], axis=-1)

    def get_hfield_x(self,snap=False):
        return self.get_array(mp.Hx, cmplx=not self.fields.is_real, snap=snap)

    def get_hfield_y(self,snap=False):
        return self.get_array(mp.Hy, cmplx=not self.fields.is_real, snap=snap)

    def get_hfield_z(self,snap=False):
        return self.get_array(mp.Hz, cmplx=not self.fields.is_real, snap=snap)

    def get_hfield_r(self,snap=False):
        return self.get_array(mp.Hr, cmplx=not self.fields.is_real, snap=snap)

    def get_hfield_p(self,snap=False):
        return self.get_array(mp.Hp, cmplx=not self.fields.is_real, snap=snap)

    def get_bfield(self,snap=False):
        if self.is_cylindrical:
            r = self.get_array(mp.Br, cmplx=not self.fields.is_real, snap=snap)
            p = self.get_array(mp.Bp, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([r, p], axis=-1)
        else:
            x = self.get_array(mp.Bx, cmplx=not self.fields.is_real, snap=snap)
            y = self.get_array(mp.By, cmplx=not self.fields.is_real, snap=snap)
            z = self.get_array(mp.Bz, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([x, y, z], axis=-1)

    def get_bfield_x(self,snap=False):
        return self.get_array(mp.Bx, cmplx=not self.fields.is_real, snap=snap)

    def get_bfield_y(self,snap=False):
        return self.get_array(mp.By, cmplx=not self.fields.is_real, snap=snap)

    def get_bfield_z(self,snap=False):
        return self.get_array(mp.Bz, cmplx=not self.fields.is_real, snap=snap)

    def get_bfield_r(self,snap=False):
        return self.get_array(mp.Br, cmplx=not self.fields.is_real, snap=snap)

    def get_bfield_p(self,snap=False):
        return self.get_array(mp.Bp, cmplx=not self.fields.is_real, snap=snap)

    def get_efield(self,snap=False):
        if self.is_cylindrical:
            r = self.get_array(mp.Er, cmplx=not self.fields.is_real, snap=snap)
            p = self.get_array(mp.Ep, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([r, p], axis=-1)
        else:
            x = self.get_array(mp.Ex, cmplx=not self.fields.is_real, snap=snap)
            y = self.get_array(mp.Ey, cmplx=not self.fields.is_real, snap=snap)
            z = self.get_array(mp.Ez, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([x, y, z], axis=-1)

    def get_efield_x(self,snap=False):
        return self.get_array(mp.Ex, cmplx=not self.fields.is_real, snap=snap)

    def get_efield_y(self,snap=False):
        return self.get_array(mp.Ey, cmplx=not self.fields.is_real, snap=snap)

    def get_efield_z(self,snap=False):
        return self.get_array(mp.Ez, cmplx=not self.fields.is_real, snap=snap)

    def get_efield_r(self,snap=False):
        return self.get_array(mp.Er, cmplx=not self.fields.is_real, snap=snap)

    def get_efield_p(self,snap=False):
        return self.get_array(mp.Ep, cmplx=not self.fields.is_real, snap=snap)

    def get_dfield(self,snap=False):
        if self.is_cylindrical:
            r = self.get_array(mp.Dr, cmplx=not self.fields.is_real, snap=snap)
            p = self.get_array(mp.Dp, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([r, p], axis=-1)
        else:
            x = self.get_array(mp.Dx, cmplx=not self.fields.is_real, snap=snap)
            y = self.get_array(mp.Dy, cmplx=not self.fields.is_real, snap=snap)
            z = self.get_array(mp.Dz, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([x, y, z], axis=-1)

    def get_dfield_x(self,snap=False):
        return self.get_array(mp.Dx, cmplx=not self.fields.is_real, snap=snap)

    def get_dfield_y(self,snap=False):
        return self.get_array(mp.Dy, cmplx=not self.fields.is_real, snap=snap)

    def get_dfield_z(self,snap=False):
        return self.get_array(mp.Dz, cmplx=not self.fields.is_real, snap=snap)

    def get_dfield_r(self,snap=False):
        return self.get_array(mp.Dr, cmplx=not self.fields.is_real, snap=snap)

    def get_dfield_p(self,snap=False):
        return self.get_array(mp.Dp, cmplx=not self.fields.is_real, snap=snap)

    def get_sfield(self,snap=False):
        if self.is_cylindrical:
            r = self.get_array(mp.Sr, cmplx=not self.fields.is_real, snap=snap)
            p = self.get_array(mp.Sp, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([r, p], axis=-1)
        else:
            x = self.get_array(mp.Sx, cmplx=not self.fields.is_real, snap=snap)
            y = self.get_array(mp.Sy, cmplx=not self.fields.is_real, snap=snap)
            z = self.get_array(mp.Sz, cmplx=not self.fields.is_real, snap=snap)
            return np.stack([x, y, z], axis=-1)

    def get_sfield_x(self,snap=False):
        return self.get_array(mp.Sx, cmplx=not self.fields.is_real, snap=snap)

    def get_sfield_y(self,snap=False):
        return self.get_array(mp.Sy, cmplx=not self.fields.is_real, snap=snap)

    def get_sfield_z(self,snap=False):
        return self.get_array(mp.Sz, cmplx=not self.fields.is_real, snap=snap)

    def get_sfield_r(self,snap=False):
        return self.get_array(mp.Sr, cmplx=not self.fields.is_real, snap=snap)

    def get_sfield_p(self,snap=False):
        return self.get_array(mp.Sp, cmplx=not self.fields.is_real, snap=snap)


    def plot2D(self, ax=None, output_plane=None, fields=None, labels=False,
               eps_parameters=None, boundary_parameters=None,
               source_parameters=None, monitor_parameters=None,
               field_parameters=None, frequency=None,
               plot_eps_flag=True, plot_sources_flag=True,
               plot_monitors_flag=True, plot_boundaries_flag=True,
               **kwargs):
        """
        Plots a 2D cross section of the simulation domain using `matplotlib`. The plot
        includes the geometry, boundary layers, sources, and monitors. Fields can also be
        superimposed on a 2D slice. Requires [matplotlib](https://matplotlib.org). Calling
        this function would look something like:

        ```py
        sim = mp.Simulation(...)
        sim.run(...)
        field_func = lambda x: 20*np.log10(np.abs(x))
        import matplotlib.pyplot as plt
        sim.plot2D(fields=mp.Ez,
                field_parameters={'alpha':0.8, 'cmap':'RdBu', 'interpolation':'none', 'post_process':field_func},
                boundary_parameters={'hatch':'o', 'linewidth':1.5, 'facecolor':'y', 'edgecolor':'b', 'alpha':0.3})
        plt.show()
        plt.savefig('sim_domain.png')
        ```

        Note: When running a [parallel simulation](Parallel_Meep.md), the `plot2D` function expects to be called
        on all processes, but only generates a plot on the master process.

        **Parameters:**

        * `ax`: a `matplotlib` axis object. `plot2D()` will add plot objects, like lines,
          patches, and scatter plots, to this object. If no `ax` is supplied, then the
          routine will create a new figure and grab its axis.
        * `output_plane`: a `Volume` object that specifies the plane over which to plot.
          Must be 2D and a subset of the grid volume (i.e., it should not extend beyond
          the cell).
        * `fields`: the field component (`mp.Ex`, `mp.Ey`, `mp.Ez`, `mp.Hx`, `mp.Hy`,
          `mp.Hz`) to superimpose over the simulation geometry. Default is `None`, where
          no fields are superimposed.
        * `labels`: if `True`, then labels will appear over each of the simulation
          elements.
        * `eps_parameters`: a `dict` of optional plotting parameters that override the
          default parameters for the geometry.
            - `interpolation='spline36'`: interpolation algorithm used to upsample the pixels.
            - `cmap='binary'`: the color map of the geometry
            - `alpha=1.0`: transparency of geometry
            - `contour=False`: if `True`, plot a contour of the geometry rather than its image
            - `contour_linewidth=1`: line width of the contour lines if `contour=True`
        * `boundary_parameters`: a `dict` of optional plotting parameters that override
          the default parameters for the boundary layers.
            - `alpha=1.0`: transparency of boundary layers
            - `facecolor='g'`: color of polygon face
            - `edgecolor='g'`: color of outline stroke
            - `linewidth=1`: line width of outline stroke
            - `hatch='\\'`: hatching pattern
        * `source_parameters`: a `dict` of optional plotting parameters that override the
          default parameters for the sources.
            - `color='r'`: color of line and pt sources
            - `alpha=1.0`: transparency of source
            - `facecolor='none'`: color of polygon face for planar sources
            - `edgecolor='r'`: color of outline stroke for planar sources
            - `linewidth=1`: line width of outline stroke
            - `hatch='\\'`: hatching pattern
            - `label_color='r'`: color of source labels
            - `label_alpha=0.3`: transparency of source label box
            - `offset=20`: distance from source center and label box
        * `monitor_parameters`: a `dict` of optional plotting parameters that override the
          default parameters for the monitors.
            - `color='g'`: color of line and point monitors
            - `alpha=1.0`: transparency of monitors
            - `facecolor='none'`: color of polygon face for planar monitors
            - `edgecolor='r'`: color of outline stroke for planar monitors
            - `linewidth=1`: line width of outline stroke
            - `hatch='\\'`: hatching pattern
            - `label_color='g'`: color of source labels
            - `label_alpha=0.3`: transparency of monitor label box
            - `offset=20`: distance from monitor center and label box
        * `field_parameters`: a `dict` of optional plotting parameters that override the
          default parameters for the fields.
            - `interpolation='spline36'`: interpolation function used to upsample field pixels
            - `cmap='RdBu'`: color map for field pixels
            - `alpha=0.6`: transparency of fields
            - `post_process=np.real`: post processing function to apply to fields (must be
              a function object)
        * `frequency`: for materials with a [frequency-dependent
          permittivity](Materials.md#material-dispersion) $\\varepsilon(f)$, specifies the
          frequency $f$ (in Meep units) of the real part of the permittivity to use in the
          plot. Defaults to the `frequency` parameter of the [Source](#source) object.

        """
        return vis.plot2D(self, ax=ax, output_plane=output_plane, fields=fields, labels=labels,
            eps_parameters=eps_parameters, boundary_parameters=boundary_parameters,
            source_parameters=source_parameters,monitor_parameters=monitor_parameters,
            field_parameters=field_parameters, frequency=frequency,
            plot_eps_flag=plot_eps_flag, plot_sources_flag=plot_sources_flag,
            plot_monitors_flag=plot_monitors_flag, plot_boundaries_flag=plot_boundaries_flag,
            **kwargs)


    def plot_fields(self,**kwargs):
        return vis.plot_fields(self,**kwargs)

    def plot3D(self):
        """
        Uses Mayavi to render a 3D simulation domain. The simulation object must be 3D.
        Can also be embedded in Jupyter notebooks.
        """
        return vis.plot3D(self)

    def visualize_chunks(self):
        """
        Displays an interactive image of how the cell is divided into chunks. Each
        rectangular region is a chunk, and each color represents a different processor.
        Requires [matplotlib](https://matplotlib.org).
        """
        vis.visualize_chunks(self)


def _create_boundary_region_from_boundary_layers(boundary_layers, gv):
    br = mp.boundary_region()

    for layer in boundary_layers:

        if isinstance(layer, Absorber):
            continue

        boundary_region_args = [
            mp.boundary_region.PML,
            layer.thickness,
            layer.R_asymptotic,
            layer.mean_stretch,
            mp.py_pml_profile,
            layer.pml_profile,
            1 / 3,
            1 / 4,
        ]

        if layer.direction == mp.ALL:
            d = mp.start_at_direction(gv.dim)
            loop_stop_directi = mp.stop_at_direction(gv.dim)

            while d < loop_stop_directi:
                if layer.side == mp.ALL:
                    b = mp.High
                    loop_stop_bi = mp.Low

                    while b != loop_stop_bi:
                        br += mp.boundary_region(*(boundary_region_args + [d, b]))
                        b = (b + 1) % 2
                        loop_stop_bi = mp.High
                else:
                    br += mp.boundary_region(*(boundary_region_args + [d, layer.side]))
                d += 1
        else:
            if layer.side == mp.ALL:
                b = mp.High
                loop_stop_bi = mp.Low

                while b != loop_stop_bi:
                    br += mp.boundary_region(*(boundary_region_args + [layer.direction, b]))
                    b = (b + 1) % 2
                    loop_stop_bi = mp.High
            else:
                br += mp.boundary_region(*(boundary_region_args + [layer.direction, layer.side]))
    return br


# Private step functions

def _combine_step_funcs(*step_funcs):
    def _combine(sim, todo):
        for func in step_funcs:
            _eval_step_func(sim, func, todo)
    return _combine


def _eval_step_func(sim, func, todo):
    num_args = get_num_args(func)

    if num_args != 1 and num_args != 2:
        raise ValueError("Step function '{}'' requires 1 or 2 arguments".format(func.__name__))
    elif num_args == 1:
        if todo == 'step':
            func(sim)
    elif num_args == 2:
        func(sim, todo)


def _when_true_funcs(cond, *step_funcs):
    def _true(sim, todo):
        if todo == 'finish' or cond(sim):
            for f in step_funcs:
                _eval_step_func(sim, f, todo)
    return _true


# Public step functions

def after_sources(*step_funcs):
    """
    Given zero or more step functions, evaluates them only for times after all of the
    sources have turned off.
    """
    def _after_sources(sim, todo):
        time = sim.fields.last_source_time()
        if sim.round_time() >= time:
            for func in step_funcs:
                _eval_step_func(sim, func, todo)
    return _after_sources


def after_sources_and_time(t, *step_funcs):
    """
    Given zero or more step functions, evaluates them only for times after all of the
    sources have turned off, plus an additional $T$ time units have elapsed.
    """
    def _after_s_and_t(sim, todo):
        time = sim.fields.last_source_time() + t - sim.round_time()
        if sim.round_time() >= time:
            for func in step_funcs:
                _eval_step_func(sim, func, todo)
    return _after_s_and_t


def after_time(t, *step_funcs):
    """
    Given zero or more step functions, evaluates them only for times after a $T$ time
    units have elapsed from the start of the run.
    """
    def _after_t(sim):
        return sim.round_time() >= t
    return _when_true_funcs(_after_t, *step_funcs)


def at_beginning(*step_funcs):
    """
    Given zero or more step functions, evaluates them only once, at the beginning of the
    run.
    """
    closure = {'done': False}

    def _beg(sim, todo):
        if not closure['done']:
            for f in step_funcs:
                _eval_step_func(sim, f, todo)
            closure['done'] = True
    return _beg


def at_end(*step_funcs):
    """
    Given zero or more step functions, evaluates them only once, at the end of the run.
    """
    def _end(sim, todo):
        if todo == 'finish':
            for func in step_funcs:
                _eval_step_func(sim, func, 'step')
            for func in step_funcs:
                _eval_step_func(sim, func, 'finish')
    return _end


def at_every(dt, *step_funcs):
    """
    Given zero or more step functions, evaluates them at every time interval of $dT$ units
    (rounded up to the next time step).
    """
    closure = {'tlast': 0.0}

    def _every(sim, todo):
        t = sim.round_time()
        if todo == 'finish' or t >= closure['tlast'] + dt + (-0.5 * sim.fields.dt):
            for func in step_funcs:
                _eval_step_func(sim, func, todo)
            closure['tlast'] = t
    return _every


def at_time(t, *step_funcs):
    """
    Given zero or more step functions, evaluates them only once, after a $T$ time units
    have elapsed from the start of the run.
    """
    closure = {'done': False}

    def _at_time(sim, todo):
        if not closure['done'] or todo == 'finish':
            for f in step_funcs:
                _eval_step_func(sim, f, todo)
        closure['done'] = closure['done'] or todo == 'step'
    return after_time(t, _at_time)


def before_time(t, *step_funcs):
    """
    Given zero or more step functions, evaluates them only for times before a $T$ time
    units have elapsed from the start of the run.
    """
    def _before_t(sim):
        return sim.round_time() < t
    return _when_true_funcs(_before_t, *step_funcs)


def during_sources(*step_funcs):
    """
    Given zero or more step functions, evaluates them only for times *before* all of the
    sources have turned off.
    """
    closure = {'finished': False}

    def _during_sources(sim, todo):
        time = sim.fields.last_source_time()
        if sim.round_time() < time:
            for func in step_funcs:
                _eval_step_func(sim, func, 'step')
        elif closure['finished'] is False:
            for func in step_funcs:
                _eval_step_func(sim, func, 'finish')
            closure['finished'] = True
    return _during_sources


def in_volume(v, *step_funcs):
    """
    Given zero or more step functions, modifies any output functions among them to only
    output a subset (or a superset) of the cell, corresponding to the `meep::volume* v`
    (created by the `Volume` function).
    """
    closure = {'cur_eps': ''}

    def _in_volume(sim, todo):
        v_save = sim.output_volume
        eps_save = sim.last_eps_filename

        sim.output_volume = sim._fit_volume_to_simulation(v).swigobj

        if closure['cur_eps']:
            sim.last_eps_filename = closure['cur_eps']
        for func in step_funcs:
            _eval_step_func(sim, func, todo)

        closure['cur_eps'] = sim.last_eps_filename
        sim.output_volume = v_save
        if eps_save:
            sim.last_eps_filename = eps_save
    return _in_volume


def in_point(pt, *step_funcs):
    """
    Given zero or more step functions, modifies any output functions among them to only
    output a single *point* of data, at `pt` (a `Vector3`).
    """
    v = Volume(pt)
    return in_volume(v, *step_funcs)


def to_appended(fname, *step_funcs):
    """
    Given zero or more step functions, modifies any output functions among them to
    *append* their data to datasets in a single newly-created file named `filename` (plus
    an `.h5` suffix and the current filename prefix). They append by adding an *extra
    dimension* to their datasets, corresponding to time.
    """
    closure = {'h5': None}

    def _to_appended(sim, todo):
        if closure['h5'] is None:
            closure['h5'] = sim.fields.open_h5file(fname, mp.h5file.WRITE, sim.get_filename_prefix())
        h5save = sim.output_append_h5
        sim.output_append_h5 = closure['h5']

        for func in step_funcs:
            _eval_step_func(sim, func, todo)

        if todo == 'finish':
            closure['h5'] = None
            sim.output_h5_hook(sim.fields.h5file_name(fname, sim.get_filename_prefix()))
        sim.output_append_h5 = h5save
    return _to_appended


def stop_when_fields_decayed(dt, c, pt, decay_by):
    """
    Return a `condition` function, suitable for passing to `Simulation.run` as the `until`
    or `until_after_sources` parameter, that examines the component `c` (e.g. `Ex`, etc.)
    at the point `pt` (a `Vector3`) and keeps running until its absolute value *squared*
    has decayed by at least `decay_by` from its maximum previous value. In particular, it
    keeps incrementing the run time by `dT` (in Meep units) and checks the maximum value
    over that time period &mdash; in this way, it won't be fooled just because the field
    happens to go through 0 at some instant.

    Note that, if you make `decay_by` very small, you may need to increase the `cutoff`
    property of your source(s), to decrease the amplitude of the small high-frequency
    components that are excited when the source turns off. High frequencies near the
    [Nyquist frequency](https://en.wikipedia.org/wiki/Nyquist_frequency) of the grid have
    slow group velocities and are absorbed poorly by [PML](Perfectly_Matched_Layer.md).
    """
    closure = {
        'max_abs': 0,
        'cur_max': 0,
        't0': 0,
    }

    def _stop(sim):
        fabs = abs(sim.get_field_point(c, pt)) * abs(sim.get_field_point(c, pt))
        closure['cur_max'] = max(closure['cur_max'], fabs)

        if sim.round_time() <= dt + closure['t0']:
            return False
        else:
            old_cur = closure['cur_max']
            closure['cur_max'] = 0
            closure['t0'] = sim.round_time()
            closure['max_abs'] = max(closure['max_abs'], old_cur)
            if closure['max_abs'] != 0 and verbosity.meep > 0:
                fmt = "field decay(t = {}): {} / {} = {}"
                print(fmt.format(sim.meep_time(), old_cur, closure['max_abs'], old_cur / closure['max_abs']))
            return old_cur <= closure['max_abs'] * decay_by
    return _stop


def stop_after_walltime(t):
    """
    Return a `condition` function, suitable for passing to `Simulation.run` as the `until`
    parameter. Stops the simulation after `t` seconds of wall time have passed.
    """
    start = mp.wall_time()
    def _stop_after_walltime(sim):
        if mp.wall_time() - start > t:
            return True
        return False
    return _stop_after_walltime


def stop_on_interrupt():
    """
    Return a `condition` function, suitable for passing to `Simulation.run` as the `until`
    parameter. Instead of terminating when receiving a SIGINT or SIGTERM signal from the
    system, the simulation will abort time stepping and continue executing any code that
    follows the `run` function (e.g., outputting fields).
    """
    shutting_down = [False]

    def _signal_handler(sig, frame):
        print("WARNING: System requested termination. Time stepping aborted.")
        shutting_down[0] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    def _stop(sim):
        return shutting_down[0]

    return _stop


def combine_step_funcs(*step_funcs):
    """
    Given zero or more step functions, return a new step function that on each step calls
    all of the passed step functions.
    """
    return _combine_step_funcs(*step_funcs)


def synchronized_magnetic(*step_funcs):
    """
    Given zero or more step functions, return a new step function that on each step calls
    all of the passed step functions with the magnetic field synchronized in time with the
    electric field. See [Synchronizing the Magnetic and Electric
    Fields](Synchronizing_the_Magnetic_and_Electric_Fields.md).
    """
    def _sync(sim, todo):
        sim.fields.synchronize_magnetic_fields()
        for f in step_funcs:
            _eval_step_func(sim, f, todo)
        sim.fields.restore_magnetic_fields()
    return _sync


def when_true(cond, *step_funcs):
    """
    Given zero or more step functions and a condition function `condition` (a function of
    no arguments), evaluate the step functions whenever `condition` returns `True`.
    """
    return _when_true_funcs(cond, *step_funcs)


def when_false(cond, *step_funcs):
    """
    Given zero or more step functions and a condition function `condition` (a function of
    no arguments), evaluate the step functions whenever `condition` returns `False`.
    """
    return _when_true_funcs(lambda: not cond, *step_funcs)


def with_prefix(pre, *step_funcs):
    """
    Given zero or more step functions, modifies any output functions among them to prepend
    the string `prefix` to the file names (much like `filename_prefix`, above).
    """
    def _with_prefix(sim, todo):
        saved_pre = sim.filename_prefix
        sim.filename_prefix = pre + sim.get_filename_prefix()

        for f in step_funcs:
            _eval_step_func(sim, f, todo)
        sim.filename_prefix = saved_pre
    return _with_prefix


def display_csv(sim, name, data):
    for d in data:
        display_run_data(sim, name, d)


def display_progress(t0, t, dt):
    t_0 = mp.wall_time()
    closure = {'tlast': mp.wall_time()}

    def _disp(sim):
        t1 = mp.wall_time()
        if t1 - closure['tlast'] >= dt:
            msg_fmt = "Meep progress: {}/{} = {:.1f}% done in {:.1f}s, {:.1f}s to go"
            val1 = sim.meep_time() - t0
            val2 = val1 / (0.01 * t)
            val3 = t1 - t_0
            val4 = (val3 * (t / val1) - val3) if val1 != 0 else 0

            if do_progress:
                sim.progress.value = val1
                sim.progress.description = "{}% done ".format(int(val2))

            if verbosity.meep > 0:
                print(msg_fmt.format(val1, t, val2, val3, val4))
            closure['tlast'] = t1

    return _disp


def data_to_str(d):
    if type(d) is complex:
        sign = '+' if d.imag >= 0 else ''
        return "{}{}{}i".format(d.real, sign, d.imag)
    else:
        return str(d)


def display_run_data(sim, data_name, data):
    if isinstance(data, Sequence):
        data_str = [data_to_str(f) for f in data]
    else:
        data_str = [data_to_str(data)]
    print("{}{}:, {}".format(data_name, sim.run_index, ', '.join(data_str)))


def convert_h5(rm_h5, convert_cmd, *step_funcs):

    def convert(fname):
        if mp.my_rank() == 0:
            cmd = convert_cmd.split()
            cmd.append(fname)
            ret = subprocess.call(cmd)
            if ret == 0 and rm_h5:
                os.remove(fname)

    def _convert_h5(sim, todo):
        hooksave = sim.output_h5_hook
        sim.output_h5_hook = convert

        for f in step_funcs:
            _eval_step_func(sim, f, todo)

        sim.output_h5_hook = hooksave

    return _convert_h5


def output_png(compnt, options, rm_h5=True):
    """
    Output the given field component (e.g. `Ex`, etc.) as a
    [PNG](https://en.wikipedia.org/wiki/PNG) image, by first outputting the HDF5 file,
    then converting to PNG via
    [h5topng](https://github.com/NanoComp/h5utils/blob/master/README.md), then deleting
    the HDF5 file. The second argument is a string giving options to pass to h5topng (e.g.
    `"-Zc bluered"`). See also [Tutorial/Basics/Output Tips and
    Tricks](Python_Tutorials/Basics.md#output-tips-and-tricks).

    It is often useful to use the h5topng `-C` or `-A` options to overlay the dielectric
    function when outputting fields. To do this, you need to know the name of the
    dielectric-function `.h5` file which must have been previously output by
    `output_epsilon`. To make this easier, a built-in shell variable `$EPS` is provided
    which refers to the last-output dielectric-function `.h5` file. So, for example
    `output_png(mp.Ez,"-C $EPS")` will output the $E_z$ field and overlay the dielectric
    contours.

    By default, `output_png` deletes the `.h5` file when it is done. To preserve the `.h5`
    file requires `output_png(component, h5topng_options, rm_h5=False)`.
    """
    closure = {'maxabs': 0.0}

    def _output_png(sim, todo):
        if todo == 'step':
            if sim.output_volume is None:
                ov = sim.fields.total_volume()
            else:
                ov = sim.output_volume

            closure['maxabs'] = max(closure['maxabs'],
                                    sim.fields.max_abs(compnt, ov))
            convert = sim.h5topng(rm_h5, "-M {} {}".format(closure['maxabs'], options),
                                  lambda sim: sim.output_component(compnt))
            convert(sim, todo)
    return _output_png


def output_epsilon(sim=None,*step_func_args,**kwargs):
    """
    Given a frequency `frequency`, (provided as a keyword argument) output $\\varepsilon$ (relative
    permittivity); for an anisotropic $\\varepsilon$ tensor the output is the [harmonic
    mean](https://en.wikipedia.org/wiki/Harmonic_mean) of the $\\varepsilon$ eigenvalues. If
    `frequency` is non-zero, the output is complex; otherwise it is the real,
    frequency-independent part of $\\varepsilon$ (the $\\omega\\to\\infty$ limit).
    When called as part of a [step function](Python_User_Interface.md#controlling-when-a-step-function-executes),
    the `sim` argument specifying the `Simulation` object can be omitted, e.g.,
    `sim.run(mp.at_beginning(mp.output_epsilon(frequency=1/0.7)),until=10)`.
    """
    if sim is None:
        return lambda sim: mp.output_epsilon(sim, *step_func_args, **kwargs)

    frequency = kwargs.pop('frequency', 0.0)
    sim.output_component(mp.Dielectric,frequency=frequency)


def output_mu(sim=None,*step_func_args,**kwargs):
    """
    Given a frequency `frequency`, (provided as a keyword argument) output $\\mu$ (relative
    permeability); for an anisotropic $\\mu$ tensor the output is the [harmonic
    mean](https://en.wikipedia.org/wiki/Harmonic_mean) of the $\\mu$ eigenvalues. If
    `frequency` is non-zero, the output is complex; otherwise it is the real,
    frequency-independent part of $\\mu$ (the $\\omega\\to\\infty$ limit).
    When called as part of a [step function](Python_User_Interface.md#controlling-when-a-step-function-executes),
    the `sim` argument specifying the `Simulation` object can be omitted, e.g.,
    `sim.run(mp.at_beginning(mp.output_mu(frequency=1/0.7)),until=10)`.
    """
    if sim is None:
        return lambda sim: mp.output_mu(sim, *step_func_args, **kwargs)

    frequency = kwargs.pop('frequency', 0.0)
    sim.output_component(mp.Permeability,frequency=frequency)


def output_hpwr(sim):
    """
    Output the magnetic-field energy density $\\mathbf{H}^* \\cdot \\mathbf{B} / 2$
    """
    sim.output_component(mp.H_EnergyDensity)


def output_dpwr(sim):
    """
    Output the electric-field energy density $\\mathbf{E}^* \\cdot \\mathbf{D} / 2$
    """
    sim.output_component(mp.D_EnergyDensity)


def output_tot_pwr(sim):
    """
    Output the total electric and magnetic energy density. Note that you might want to
    wrap this step function in `synchronized_magnetic` to compute it more accurately. See
    [Synchronizing the Magnetic and Electric
    Fields](Synchronizing_the_Magnetic_and_Electric_Fields.md).
    """
    sim.output_component(mp.EnergyDensity)


def output_hfield(sim):
    """
    Outputs *all* the components of the field *h*, (magnetic) to an HDF5 file. That is,
    the different components are stored as different datasets within the *same* file.
    """
    sim.output_components('h', mp.Hx, mp.Hy, mp.Hz, mp.Hr, mp.Hp)


def output_hfield_x(sim):
    """
    Output the $x$ component of the field *h* (magnetic). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Hx)


def output_hfield_y(sim):
    """
    Output the $y$ component of the field *h* (magnetic). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Hy)


def output_hfield_z(sim):
    """
    Output the $z$ component of the field *h* (magnetic). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Hz)


def output_hfield_r(sim):
    """
    Output the $r$ component of the field *h* (magnetic). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Hr)


def output_hfield_p(sim):
    """
    Output the $\\phi$ component of the field *h* (magnetic). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively.
    """
    sim.output_component(mp.Hp)


def output_bfield(sim):
    """
    Outputs *all* the components of the field *b*, (magnetic) to an HDF5 file. That is,
    the different components are stored as different datasets within the *same* file.
    """
    sim.output_components('b', mp.Bx, mp.By, mp.Bz, mp.Br, mp.Bp)


def output_bfield_x(sim):
    """
    Output the $x$ component of the field *b* (magnetic). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Bx)


def output_bfield_y(sim):
    """
    Output the $y$ component of the field *b* (magnetic). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.By)


def output_bfield_z(sim):
    """
    Output the $z$ component of the field *b* (magnetic). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Bz)


def output_bfield_r(sim):
    """
    Output the $r$ component of the field *b* (magnetic). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Br)


def output_bfield_p(sim):
    """
    Output the $\\phi$ component of the field *b* (magnetic). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively. Note that for outputting the Poynting flux, you
    might want to wrap the step function in `synchronized_magnetic` to compute it more
    accurately. See [Synchronizing the Magnetic and Electric
    Fields](Synchronizing_the_Magnetic_and_Electric_Fields.md).
    """
    sim.output_component(mp.Bp)


def output_efield(sim):
    """
    Outputs *all* the components of the field *e*, (electric) to an HDF5 file. That is,
    the different components are stored as different datasets within the *same* file.
    """
    sim.output_components('e', mp.Ex, mp.Ey, mp.Ez, mp.Er, mp.Ep)


def output_efield_x(sim):
    """
    Output the $x$ component of the field *e* (electric). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Ex)


def output_efield_y(sim):
    """
    Output the $y$ component of the field *e* (electric). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Ey)


def output_efield_z(sim):
    """
    Output the $z$ component of the field *e* (electric). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Ez)


def output_efield_r(sim):
    """
    Output the $r$ component of the field *e* (electric). If the field is complex, outputs
    two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real and
    imaginary parts, respectively.
    """
    sim.output_component(mp.Er)


def output_efield_p(sim):
    """
    Output the $\\phi$ component of the field *e* (electric). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively. Note that for outputting the Poynting flux, you
    might want to wrap the step function in `synchronized_magnetic` to compute it more
    accurately. See [Synchronizing the Magnetic and Electric
    Fields](Synchronizing_the_Magnetic_and_Electric_Fields.md).
    """
    sim.output_component(mp.Ep)


def output_dfield(sim):
    """
    Outputs *all* the components of the field *d*, (displacement) to an HDF5 file. That
    is, the different components are stored as different datasets within the *same* file.
    """
    sim.output_components('d', mp.Dx, mp.Dy, mp.Dz, mp.Dr, mp.Dp)


def output_dfield_x(sim):
    """
    Output the $x$ component of the field *d* (displacement). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively.
    """
    sim.output_component(mp.Dx)


def output_dfield_y(sim):
    """
    Output the $y$ component of the field *d* (displacement). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively.
    """
    sim.output_component(mp.Dy)


def output_dfield_z(sim):
    """
    Output the $z$ component of the field *d* (displacement). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively.
    """
    sim.output_component(mp.Dz)


def output_dfield_r(sim):
    """
    Output the $r$ component of the field *d* (displacement). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively.
    """
    sim.output_component(mp.Dr)


def output_dfield_p(sim):
    """
    Output the $\\phi$ component of the field *d* (displacement). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively. Note that for outputting the Poynting flux, you
    might want to wrap the step function in `synchronized_magnetic` to compute it more
    accurately. See [Synchronizing the Magnetic and Electric
    Fields](Synchronizing_the_Magnetic_and_Electric_Fields.md).
    """
    sim.output_component(mp.Dp)


# MPB compatibility
def output_poynting(sim):
    """
    Output the Poynting flux $\\Re [\\mathbf{E}^* \\times \\mathbf{H}]$. Note that you
    might want to wrap this step function in `synchronized_magnetic` to compute it more
    accurately. See [Synchronizing the Magnetic and Electric
    Fields](Synchronizing_the_Magnetic_and_Electric_Fields.md).
    """
    sim.output_components('s', mp.Sx, mp.Sy, mp.Sz, mp.Sr, mp.Sp)


def output_poynting_x(sim):
    sim.output_component(mp.Sx)


def output_poynting_y(sim):
    sim.output_component(mp.Sy)


def output_poynting_z(sim):
    sim.output_component(mp.Sz)


def output_poynting_r(sim):
    sim.output_component(mp.Sr)


def output_poynting_p(sim):
    sim.output_component(mp.Sp)


def output_sfield(sim):
    """
    Outputs *all* the components of the field *s*, (poynting flux) to an HDF5 file. That
    is, the different components are stored as different datasets within the *same* file.
    Note that you might want to wrap this step function in `synchronized_magnetic` to
    compute it more accurately. See [Synchronizing the Magnetic and Electric
    Fields](Synchronizing_the_Magnetic_and_Electric_Fields.md).
    """
    sim.output_components('s', mp.Sx, mp.Sy, mp.Sz, mp.Sr, mp.Sp)


def output_sfield_x(sim):
    """
    Output the $x$ component of the field *s* (poynting flux). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively.
    """
    sim.output_component(mp.Sx)


def output_sfield_y(sim):
    """
    Output the $y$ component of the field *s* (poynting flux). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively.
    """
    sim.output_component(mp.Sy)


def output_sfield_z(sim):
    """
    Output the $z$ component of the field *s* (poynting flux). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively.
    """
    sim.output_component(mp.Sz)


def output_sfield_r(sim):
    """
    Output the $r$ component of the field *s* (poynting flux). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively.
    """
    sim.output_component(mp.Sr)


def output_sfield_p(sim):
    """
    Output the $\\phi$ component of the field *s* (poynting flux). If the field is complex,
    outputs two datasets, e.g. `ex.r` and `ex.i`, within the same HDF5 file for the real
    and imaginary parts, respectively. Note that for outputting the Poynting flux, you
    might want to wrap the step function in `synchronized_magnetic` to compute it more
    accurately. See [Synchronizing the Magnetic and Electric
    Fields](Synchronizing_the_Magnetic_and_Electric_Fields.md).
    """
    sim.output_component(mp.Sp)


def Ldos(*args):
    """
    `Ldos(fcen, df, nfreq, freq)`  ##sig

    Create an LDOS object with either frequency bandwidth `df` centered at `fcen` and
    `nfreq` equally spaced frequency points or an array/list `freq` for arbitrarily spaced
    frequencies. This can be passed to the `dft_ldos` step function below as a keyword
    argument.
    """
    args = fix_dft_args(args, 0)
    freq = args[0]

    return mp._dft_ldos(freq)


def dft_ldos(*args, **kwargs):
    """
    `dft_ldos(fcen=None, df=None, nfreq=None, freq=None, ldos=None)`   ##sig

    Compute the power spectrum of the sources (usually a single point dipole source),
    normalized to correspond to the LDOS, in either a frequency bandwidth `df` centered at
    `fcen` and `nfreq` equally spaced frequency points or an array/list `freq` for
    arbitrarily spaced frequencies. One can also pass in an `Ldos` object as
    `dft_ldos(ldos=my_Ldos)`.

    The resulting spectrum is outputted as comma-delimited text, prefixed by `ldos:,`, and
    is also stored in the `ldos_data` variable of the `Simulation` object after the `run`
    is complete.
    """
    ldos = kwargs.get('ldos', None)
    if ldos is None:
        args = fix_dft_args(args, 0)
        freq = args[0]
        if isinstance(freq, (np.ndarray,list)):
            ldos = mp._dft_ldos(freq)
        else:
            raise TypeError("dft_ldos only accepts freq_min,freq_max,nfreq (3 numbers) or freq (array/list) or ldos (keyword argument)")

    def _ldos(sim, todo):
        if todo == 'step':
            ldos.update(sim.fields)
        else:
            sim.ldos_data = mp._dft_ldos_ldos(ldos)
            sim.ldos_Fdata = mp._dft_ldos_F(ldos)
            sim.ldos_Jdata = mp._dft_ldos_J(ldos)
            display_csv(sim, 'ldos', zip(mp.get_ldos_freqs(ldos), sim.ldos_data))
    return _ldos


def scale_flux_fields(s, flux):
    """
    Scale the Fourier-transformed fields in `flux` by the complex number `s`. e.g.
    `load_minus_flux` is equivalent to `load_flux` followed by `scale_flux_fields` with
    `s=-1`.
    """
    flux.scale_dfts(s)


def get_ldos_freqs(l):
    """
    Given an LDOS object, returns a list of the frequencies that it is computing the
    spectrum for.
    """
    return [l.freq[i] for i in range(l.freq.size())]


def get_flux_freqs(f):
    """
    Given a flux object, returns a list of the frequencies that it is computing the
    spectrum for.
    """
    return [f.freq[i] for i in range(f.freq.size())]


def get_fluxes(f):
    """
    Given a flux object, returns a list of the current flux spectrum that it has
    accumulated.
    """
    return f.flux()


def scale_force_fields(s, force):
    force.scale_dfts(s)


def get_eigenmode_freqs(f):
    """
    Given a flux object, returns a list of the frequencies that it is computing the
    spectrum for.
    """
    return [f.freq[i] for i in range(f.freq.size())]


def get_force_freqs(f):
    """
    Given a force object, returns a list of the frequencies that it is computing the
    spectrum for.
    """
    return [f.freq[i] for i in range(f.freq.size())]


def get_forces(f):
    """
    Given a force object, returns a list of the current force spectrum that it has
    accumulated.
    """
    return f.force()


def scale_near2far_fields(s, near2far):
    """
    Scale the Fourier-transformed fields in `near2far` by the complex number `s`. e.g.
    `load_minus_near2far` is equivalent to `load_near2far` followed by
    `scale_near2far_fields` with `s=-1`.
    """
    near2far.scale_dfts(s)


def get_near2far_freqs(f):
    """
    Given a `near2far` object, returns a list of the frequencies that it is computing the
    spectrum for.
    """
    return [f.freq[i] for i in range(f.freq.size())]


def scale_energy_fields(s, ef):
    df.scale_dfts(s)


def get_energy_freqs(f):
    """
    Given an energy object, returns a list of the frequencies that it is computing the
    spectrum for.
    """
    return [f.freq[i] for i in range(f.freq.size())]


def get_electric_energy(f):
    """
    Given an energy object, returns a list of the current energy density spectrum for the
    electric fields that it has accumulated.
    """
    return f.electric()


def get_magnetic_energy(f):
    """
    Given an energy object, returns a list of the current energy density spectrum for the
    magnetic fields that it has accumulated.
    """
    return f.magnetic()


def get_total_energy(f):
    """
    Given an energy object, returns a list of the current energy density spectrum for the
    total fields that it has accumulated.
    """
    return f.total()


def interpolate(n, nums):
    """
    Given a list of numbers or `Vector3`s as `nums`, linearly interpolates between them to
    add `n` new evenly-spaced values between each pair of consecutive values in the
    original list.
    """
    res = []
    if isinstance(nums[0], mp.Vector3):
        for low, high in zip(nums, nums[1:]):
            x = np.linspace(low.x, high.x, n + 1, endpoint=False).tolist()
            y = np.linspace(low.y, high.y, n + 1, endpoint=False).tolist()
            z = np.linspace(low.z, high.z, n + 1, endpoint=False).tolist()

            for i in range(len(x)):
                res.append(mp.Vector3(x[i], y[i], z[i]))
    else:
        for low, high in zip(nums, nums[1:]):
            res.extend(np.linspace(low, high, n + 1, endpoint=False).tolist())

    return res + [nums[-1]]


# extract center and size of a meep::volume
def get_center_and_size(vol):
    """
    Utility function that takes a `meep::volume` `vol` and returns the center and size of
    the volume as a tuple of `Vector3`.
    """
    rmin = vol.get_min_corner()
    rmax = vol.get_max_corner()
    v3rmin = mp.Vector3(rmin.x(), rmin.y(), rmin.z())
    v3rmax = mp.Vector3(rmax.x(), rmax.y(), rmax.z())

    if vol.dim == mp.D2:
        v3rmin.z = 0
        v3rmax.z = 0
    elif vol.dim == mp.D1:
        v3rmin.x = 0
        v3rmin.y = 0
        v3rmin.y = 0
        v3rmax.y = 0

    center = 0.5 * (v3rmin + v3rmax)
    size = v3rmax - v3rmin
    return center, size

def GDSII_layers(fname):
    """
    Returns a list of integer-valued layer indices for the layers present in
    the specified GDSII file.

    ```python
    mp.GDSII_layers('python/examples/coupler.gds')
    Out[2]: [0, 1, 2, 3, 4, 5, 31, 32]
    ```
    """
    return list(mp.get_GDSII_layers(fname))

def GDSII_vol(fname, layer, zmin, zmax):
    """
    Returns a `mp.Volume` read from a GDSII file `fname` on layer number `layer` with
    `zmin` and `zmax` (default 0). This function is useful for creating a `FluxRegion`
    from a GDSII file as follows:

    ```python
    fr = mp.FluxRegion(volume=mp.GDSII_vol(fname, layer, zmin, zmax))
    ```
    """
    meep_vol = mp.get_GDSII_volume(fname, layer, zmin, zmax)
    dims = meep_vol.dim + 1
    is_cyl = False

    if dims == 4:
        # cylindrical
        dims = 2
        is_cyl = True

    center, size = get_center_and_size(meep_vol)

    return Volume(center, size, dims, is_cyl)


def GDSII_prisms(material, fname, layer=-1, zmin=0.0, zmax=0.0):
    """
    Returns a list of `GeometricObject`s with `material` (`mp.Medium`) on layer number
    `layer` of a GDSII file `fname` with `zmin` and `zmax` (default 0).
    """
    return mp.get_GDSII_prisms(material, fname, layer, zmin, zmax)


def complexarray(re, im):
    z = im * 1j
    z += re
    return z

def quiet(quietval=True):
    """
    Meep ordinarily prints various diagnostic and progress information to standard output.
    This output can be suppressed by calling this function with `True` (the default). The
    output can be enabled again by passing `False`. This sets a global variable, so the
    value will persist across runs within the same script.

    This function is deprecated, please use the [Verbosity](#verbosity) class instead.
    """
    verbosity(int(not quietval))
    warnings.warn("quiet has been deprecated; use the Verbosity class instead", RuntimeWarning)


def get_num_groups():
    # Lazy import
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    return comm.allreduce(int(mp.my_rank() == 0), op=MPI.SUM)

def get_group_masters():
    # Lazy import
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()
    num_groups = mp.get_num_groups

    # Check if current worker is a group master
    is_group_master = True if mp.my_rank() == 0 else False
    group_master_idx = np.zeros((num_workers,),dtype=np.bool)

    # Formulate send and receive packets
    smsg = [np.array([is_group_master]),([1]*num_workers, [0]*num_workers)]
    rmsg = [group_master_idx,([1]*num_workers, list(range(num_workers)))]

    # Send and receive
    comm.Alltoallv(smsg, rmsg)

    # get rank of each group master
    group_masters = np.arange(num_workers)[group_master_idx] # rank index of each group leader

    return group_masters

def merge_subgroup_data(data):
    # Lazy import
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()
    num_groups = get_num_groups()

    # Initialize new input and output datasets
    input=np.array(data,copy=True,order='F')
    shape=input.shape
    size=input.size
    out_shape=shape + (num_groups,)
    output=np.zeros(out_shape,input.dtype,order='F')

    # Get group masters
    group_masters = get_group_masters()

    # Specify how much talking each proc will do. Only group masters send data.
    if mp.my_rank() == 0:
        scount = np.array([size] * num_workers)
    else:
        scount = np.array([0] * num_workers)
    rcount = np.array([0] * num_workers)
    rcount[group_masters] = size

    # Specify array mapping
    sdsp = [0] * num_workers
    rdsp = [0] * num_workers
    buf_idx = 0
    for grpidx in group_masters:
        rdsp[grpidx] = buf_idx # offset group leader worker by size of each count
        buf_idx += size

    # Formulate send and receive packets
    smsg = [input, (scount, sdsp)]
    rmsg = [output, (rcount, rdsp)]

    # Send and receive
    comm.Alltoallv(smsg, rmsg)

    return output

class BinaryPartition(object):
    """
    Binary tree class used for specifying a cell partition of arbitrary sized chunks for use as the
    `chunk_layout` parameter of the `Simulation` class object.
    """
    def __init__(self, data=None, split_dir=None, split_pos=None, left=None, right=None, proc_id=None):
        """
        The constructor accepts three separate groups of arguments: (1) `data`: a list of lists where each
        list entry is either (a) a node defined as `[ (split_dir,split_pos), left, right ]` for which `split_dir`
        and `split_pos` define the splitting direction (i.e., `mp.X`, `mp.Y`, `mp.Z`) and position (e.g., `3.5`,
        `-4.2`, etc.) and `left` and `right` are the two branches (themselves `BinaryPartition` objects)
        or (b) a leaf with integer value for the process ID `proc_id` in the range between 0 and number of processes
        - 1 (inclusive), (2) a node defined using `split_dir`, `split_pos`, `left`, and `right`, or (3) a leaf with
        `proc_id`. Note that the same process ID can be assigned to as many chunks as you want, which means that one
        process timesteps multiple chunks. If you use fewer MPI processes, then the process ID is taken modulo the number
        of processes.
        """
        self.split_dir = None
        self.split_pos = None
        self.proc_id = None
        self.left = None
        self.right = None
        if data is not None:
            if isinstance(data,list) and len(data) == 3:
                if isinstance(data[0],tuple) and len(data[0]) == 2:
                    self.split_dir = data[0][0]
                    self.split_pos = data[0][1]
                else:
                    raise ValueError("expecting 2-tuple (split_dir,split_pos) but got {}".format(data[0]))
                self.left = BinaryPartition(data=data[1])
                self.right = BinaryPartition(data=data[2])
            elif isinstance(data,int):
                self.proc_id = data
            else:
                raise ValueError("expecting list [(split_dir,split_pos), left, right] or int (proc_id) but got {}".format(data))
        elif split_dir is not None:
            self.split_dir = split_dir
            self.split_pos = split_pos
            self.left = left
            self.right = right
        else:
            self.proc_id = proc_id
    
    def print(self):
        """Pretty-prints the tree structure of the BinaryPartition object."""
        print(str(self) + " with {} chunks:".format(self.numchunks()))
        for line in self._print(is_root=True):
            print(line)

    def _print(self, prefix="", is_root=True):
        # pointers
        ptr_l = ' ├L─ '
        ext_l = ' │   '
        ptr_r = ' └R─ '
        ext_r = '     '

        if is_root:
            yield prefix + self._node_info()

        if self.left is not None and self.right is not None:
            yield prefix + ptr_l + self.left._node_info()
            if self.left.left is not None and self.left.right is not None:
                yield from self.left._print(prefix=prefix+ext_l, is_root=False)

            yield prefix + ptr_r + self.right._node_info()
            if self.right.left is not None and self.right.right is not None:
                yield from self.right._print(prefix=prefix+ext_r, is_root=False)

    def _node_info(self) -> str:
        if self.proc_id is not None:
            return "<proc_id={}>".format(self.proc_id)
        else:
            split_dir_str = {
                mp.X: "X",
                mp.Y: "Y",
                mp.Z: "Z"
            }[self.split_dir]
            return "<split_dir={}, split_pos={}>".format(
                split_dir_str, self.split_pos)

    def _numchunks(self,bp):
        if bp is None:
            return 0
        return max(self._numchunks(bp.left)+self._numchunks(bp.right), 1)

    def numchunks(self):
        return self._numchunks(self)
