"""Handling of objective functions and objective quantities."""

import abc
import numpy as np
import meep as mp
from .filter_source import FilteredSource
from .optimization_problem import Grid
from meep.simulation import py_v3_to_vec


class ObjectiveQuantity(abc.ABC):
    """A differentiable objective quantity.

    Attributes:
        sim: the Meep simulation object with which the objective quantity is registered.
        frequencies: the frequencies at which the objective quantity is evaluated.
        num_freq: the number of frequencies at which the objective quantity is evaluated.
    """
    def __init__(self, sim):
        self.sim = sim
        self._eval = None
        self._frequencies = None

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def num_freq(self):
        return len(self.frequencies)

    @abc.abstractmethod
    def __call__(self):
        """Evaluates the objective quantity."""

    @abc.abstractmethod
    def register_monitors(self, frequencies):
        """Registers monitors in the forward simulation."""

    @abc.abstractmethod
    def place_adjoint_source(self, dJ):
        """Places appropriate sources for the adjoint simulation."""

    def get_evaluation(self):
        """Evaluates the objective quantity."""
        if self._eval:
            return self._eval
        else:
            raise RuntimeError(
                'You must first run a forward simulation before requesting the evaluation of an objective quantity.'
            )

    def _adj_src_scale(self, include_resolution=True):
        """Calculates the scale for the adjoint sources."""
        T = self.sim.meep_time()
        dt = self.sim.fields.dt
        src = self._create_time_profile()

        if include_resolution:
            num_dims = self.sim._infer_dimensions(self.sim.k_point)
            dV = 1 / self.sim.resolution**num_dims
        else:
            dV = 1

        iomega = (1.0 - np.exp(-1j * (2 * np.pi * self._frequencies) * dt)) * (
            1.0 / dt
        )  # scaled frequency factor with discrete time derivative fix

        # an ugly way to calcuate the scaled dtft of the forward source
        y = np.array([src.swigobj.current(t, dt)
                      for t in np.arange(0, T, dt)])  # time domain signal
        fwd_dtft = np.matmul(
            np.exp(1j * 2 * np.pi * self._frequencies[:, np.newaxis] *
                   np.arange(y.size) * dt), y) * dt / np.sqrt(
                       2 * np.pi)  # dtft

        # Interestingly, the real parts of the DTFT and fourier transform match, but the imaginary parts are very different...
        #fwd_dtft = src.fourier_transform(src.frequency)
        '''
        For some reason, there seems to be an additional phase
        factor at the center frequency that needs to be applied
        to *all* frequencies...
        '''
        src_center_dtft = np.matmul(
            np.exp(1j * 2 * np.pi * np.array([src.frequency])[:, np.newaxis] *
                   np.arange(y.size) * dt), y) * dt / np.sqrt(2 * np.pi)
        adj_src_phase = np.exp(1j * np.angle(src_center_dtft))

        if self._frequencies.size == 1:
            # Single frequency simulations. We need to drive it with a time profile.
            scale = dV * iomega / fwd_dtft / adj_src_phase  # final scale factor
        else:
            # multi frequency simulations
            scale = dV * iomega / adj_src_phase
        return scale

    def _create_time_profile(self, fwidth_frac=0.1):
        """Creates a time domain waveform for normalizing the adjoint source(s).

        For single frequency objective functions, we should generate a guassian pulse with a reasonable
        bandwidth centered at said frequency.

        TODO:
        The user may specify a scalar valued objective function across multiple frequencies (e.g. MSE) in
        which case we should check that all the frequencies fit in the specified bandwidth.
        """
        return mp.GaussianSource(
            np.mean(self._frequencies),
            fwidth=fwidth_frac * np.mean(self._frequencies),
        )


class EigenmodeCoefficient(ObjectiveQuantity):
    def __init__(self,
                 sim,
                 volume,
                 mode,
                 forward=True,
                 kpoint_func=None,
                 **kwargs):
        super().__init__(sim)
        self.volume = volume
        self.mode = mode
        self.forward = forward
        self.kpoint_func = kpoint_func
        self.eigenmode_kwargs = kwargs
        self._monitor = None
        self._normal_direction = None
        self._cscale = None

    def register_monitors(self, frequencies):
        self._frequencies = np.asarray(frequencies)
        self._monitor = self.sim.add_mode_monitor(
            frequencies,
            mp.ModeRegion(center=self.volume.center, size=self.volume.size),
            yee_grid=True,
        )
        self._normal_direction = self._monitor.normal_direction
        return self._monitor

    def place_adjoint_source(self, dJ):
        dJ = np.atleast_1d(dJ)
        direction_scalar = -1 if self.forward else 1
        time_src = self._create_time_profile()
        if self.kpoint_func is None:
            if self._normal_direction == 0:
                k0 = direction_scalar * mp.Vector3(x=1)
            elif self._normal_direction == 1:
                k0 = direction_scalar * mp.Vector3(y=1)
            elif self._normal_direction == 2:
                k0 = direction_scalar * mp.Vector3(z=1)
        else:
            k0 = direction_scalar * self.kpoint_func(time_src.frequency, 1)
        if dJ.ndim == 2:
            dJ = np.sum(dJ, axis=1)
        da_dE = 0.5 * self._cscale  # scalar popping out of derivative

        scale = self._adj_src_scale()

        if self._frequencies.size == 1:
            amp = da_dE * dJ * scale
            src = time_src
        else:
            scale = da_dE * dJ * scale
            src = FilteredSource(
                time_src.frequency,
                self._frequencies,
                scale,
                self.sim.fields.dt,
            )
            amp = 1

        source = mp.EigenModeSource(
            src,
            eig_band=self.mode,
            direction=mp.NO_DIRECTION,
            eig_kpoint=k0,
            amplitude=amp,
            eig_match_freq=True,
            size=self.volume.size,
            center=self.volume.center,
            **self.eigenmode_kwargs,
        )
        return [source]

    def __call__(self):
        direction = mp.NO_DIRECTION if self.kpoint_func else mp.AUTOMATIC
        ob = self.sim.get_eigenmode_coefficients(
            self._monitor,
            [self.mode],
            direction=direction,
            kpoint_func=self.kpoint_func,
            **self.eigenmode_kwargs,
        )
        # record eigenmode coefficients for scaling
        self._eval = np.squeeze(ob.alpha[:, :, int(not self.forward)])
        self._cscale = ob.cscale  # pull scaling factor
        return self._eval


class FourierFields(ObjectiveQuantity):
    def __init__(self, sim, volume, component):
        super().__init__(sim)
        self.volume = volume
        self.component = component

    def register_monitors(self, frequencies):
        self._frequencies = np.asarray(frequencies)
        self._monitor = self.sim.add_dft_fields(
            [self.component],
            self._frequencies,
            where=self.volume,
            yee_grid=False,
        )
        return self._monitor

    def place_adjoint_source(self, dJ):
        time_src = self._create_time_profile()
        sources = []
        scale = self._adj_src_scale()

        x_dim, y_dim, z_dim = len(self._dg.x), len(self._dg.y), len(self._dg.z)

        if self.num_freq == 1:
            amp = -dJ[0].copy().reshape(x_dim, y_dim, z_dim) * scale
            if self.component in [mp.Hx, mp.Hy, mp.Hz]:
                amp = -amp
            for zi in range(z_dim):
                for yi in range(y_dim):
                    for xi in range(x_dim):
                        if amp[xi, yi, zi] != 0:
                            sources += [
                                mp.Source(
                                    time_src,
                                    component=self.component,
                                    amplitude=amp[xi, yi, zi],
                                    center=mp.Vector3(self._dg.x[xi],
                                                      self._dg.y[yi],
                                                      self._dg.z[zi]),
                                )
                            ]
        else:
            '''The adjoint solver requires the objective function
            to be scalar valued with regard to objective arguments
            and position, but the function may be vector valued
            with regard to frequency. In this case, the Jacobian
            will be of the form [F,F,...] where F is the number of
            frequencies. Because of linearity, we can sum across the
            second frequency dimension to calculate a frequency
            scale factor for each point (rather than a scale vector).
            '''
            dJ = np.sum(
                dJ,
                axis=1)  # sum along first dimension bc Jacobian is always diag
            dJ_4d = np.array([
                dJ[f].copy().reshape(x_dim, y_dim, z_dim)
                for f in range(self.num_freq)
            ])
            if self.component in [mp.Hx, mp.Hy, mp.Hz]:
                dJ_4d = -dJ_4d
            for zi in range(z_dim):
                for yi in range(y_dim):
                    for xi in range(x_dim):
                        '''We only need to add a current source if the
                        jacobian is nonzero for all frequencies at 
                        that particular point. Otherwise, the fitting
                        algorithm is going to fail.
                        '''
                        if not np.all((dJ_4d[:, xi, yi, zi] == 0)):
                            final_scale = -dJ_4d[:, xi, yi, zi] * scale
                            src = FilteredSource(
                                time_src.frequency,
                                self._frequencies,
                                final_scale,
                                self.sim.fields.dt,
                            )
                            sources += [
                                mp.Source(
                                    src,
                                    component=self.component,
                                    amplitude=1,
                                    center=mp.Vector3(self._dg.x[xi],
                                                      self._dg.y[yi],
                                                      self._dg.z[zi]),
                                )
                            ]

        return sources

    def __call__(self):
        self._dg = Grid(*self.sim.get_array_metadata(dft_cell=self._monitor))
        self._eval = np.array([
            self.sim.get_dft_array(self._monitor, self.component, i)
            for i in range(self.num_freq)
        ])
        return self._eval


class Near2FarFields(ObjectiveQuantity):
    def __init__(self, sim, Near2FarRegions, far_pts):
        super().__init__(sim)
        self.Near2FarRegions = Near2FarRegions
        self.far_pts = far_pts  #list of far pts
        self._nfar_pts = len(far_pts)

    def register_monitors(self, frequencies):
        self._frequencies = np.asarray(frequencies)
        self._monitor = self.sim.add_near2far(
            self._frequencies,
            *self.Near2FarRegions,
            yee_grid=True,
        )
        return self._monitor

    def place_adjoint_source(self, dJ):
        time_src = self._create_time_profile()
        sources = []
        if dJ.ndim == 4:
            dJ = np.sum(dJ, axis=0)
        dJ = dJ.flatten()
        farpt_list = np.array([list(pi) for pi in self.far_pts]).flatten()
        far_pt0 = self.far_pts[0]
        far_pt_vec = py_v3_to_vec(self.sim.dimensions, far_pt0,
                                  self.sim.is_cylindrical)

        all_nearsrcdata = self._monitor.swigobj.near_sourcedata(
            far_pt_vec, farpt_list, self._nfar_pts, dJ)
        for near_data in all_nearsrcdata:
            cur_comp = near_data.near_fd_comp
            amp_arr = np.array(near_data.amp_arr).reshape(-1, self.num_freq)
            scale = amp_arr * self._adj_src_scale(include_resolution=False)

            if self.num_freq == 1:
                sources += [mp.IndexedSource(time_src, near_data, scale[:, 0])]
            else:
                src = FilteredSource(
                    time_src.frequency,
                    self._frequencies,
                    scale,
                    dt,
                )
                (num_basis, num_pts) = src.nodes.shape
                for basis_i in range(num_basis):
                    sources += [
                        mp.IndexedSource(
                            src.time_src_bf[basis_i],
                            near_data,
                            src.nodes[basis_i],
                        )
                    ]

        return sources

    def __call__(self):
        self._eval = np.array([
            self.sim.get_farfield(self._monitor, far_pt)
            for far_pt in self.far_pts
        ]).reshape((self._nfar_pts, self.num_freq, 6))
        return self._eval
