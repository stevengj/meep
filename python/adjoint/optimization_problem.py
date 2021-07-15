import meep as mp
import numpy as np
from jax import grad, jacobian
from collections import namedtuple

Grid = namedtuple('Grid', ['x', 'y', 'z', 'w'])
YeeDims = namedtuple('YeeDims', ['Ex', 'Ey', 'Ez'])


class DesignRegion(object):
    def __init__(self,
                 design_parameters,
                 volume=None,
                 size=None,
                 center=mp.Vector3(),
                 MaterialGrid=None):
        self.volume = volume if volume else mp.Volume(center=center, size=size)
        self.size = self.volume.size
        self.center = self.volume.center
        self.design_parameters = design_parameters
        self.num_design_params = design_parameters.num_params
        self.MaterialGrid = MaterialGrid

    def update_design_parameters(self, design_parameters):
        self.design_parameters.update_weights(design_parameters)

    def get_gradient(self, sim, fields_a, fields_f, frequencies):
        for c in range(3):
            fields_a[c] = fields_a[c].flatten(order='C')
            fields_f[c] = fields_f[c].flatten(order='C')
        fields_a = np.concatenate(fields_a)
        fields_f = np.concatenate(fields_f)
        num_freqs = np.array(frequencies).size

        grad = np.zeros((num_freqs, self.num_design_params))  # preallocate

        geom_list = sim.geometry
        f = sim.fields
        vol = sim._fit_volume_to_simulation(self.volume)
        # compute the gradient
        mp._get_gradient(grad, fields_a, fields_f, vol, np.array(frequencies),
                         geom_list, f)

        return np.squeeze(grad).T


class OptimizationProblem(object):
    """Top-level class in the MEEP adjoint module.

    Intended to be instantiated from user scripts with mandatory constructor
    input arguments specifying the data required to define an adjoint-based
    optimization.

    The class knows how to do one basic thing: Given an input vector
    of design variables, compute the objective function value (forward
    calculation) and optionally its gradient (adjoint calculation).
    This is done by the __call__ method.

    """
    def __init__(self,
                 simulation,
                 objective_functions,
                 objective_arguments,
                 design_regions,
                 frequencies=None,
                 fcen=None,
                 df=None,
                 nf=None,
                 decay_dt=50,
                 decay_fields=[mp.Ez],
                 decay_by=1e-6,
                 minimum_run_time=0,
                 maximum_run_time=None):

        self.sim = simulation

        if isinstance(objective_functions, list):
            self.objective_functions = objective_functions
        else:
            self.objective_functions = [objective_functions]
        self.objective_arguments = objective_arguments
        self.f_bank = []  # objective function evaluation history

        if isinstance(design_regions, list):
            self.design_regions = design_regions
        else:
            self.design_regions = [design_regions]

        self.num_design_params = [
            ni.num_design_params for ni in self.design_regions
        ]
        self.num_design_regions = len(self.design_regions)

        # TODO typecheck frequency choices
        if frequencies is not None:
            self.frequencies = frequencies
            self.nf = np.array(frequencies).size
        else:
            if nf == 1:
                self.nf = nf
                self.frequencies = [fcen]
            else:
                fmax = fcen + 0.5 * df
                fmin = fcen - 0.5 * df
                dfreq = (fmax - fmin) / (nf - 1)
                self.frequencies = np.linspace(fmin,
                                               fmin + dfreq * nf,
                                               num=nf,
                                               endpoint=False)
                self.nf = nf

        if self.nf == 1:
            self.fcen_idx = 0
        else:
            self.fcen_idx = int(
                np.argmin(
                    np.abs(
                        np.asarray(self.frequencies) -
                        np.mean(np.asarray(self.frequencies)))**
                    2))  # index of center frequency

        self.decay_by = decay_by
        self.decay_fields = decay_fields
        self.decay_dt = decay_dt
        self.minimum_run_time = minimum_run_time
        self.maximum_run_time = maximum_run_time

        # store sources for finite difference estimations
        self.forward_sources = self.sim.sources

        # The optimizer has three allowable states : "INIT", "FWD", and "ADJ".
        #    INIT - The optimizer is initialized and ready to run a forward simulation
        #    FWD  - The optimizer has already run a forward simulation
        #    ADJ  - The optimizer has already run an adjoint simulation (but not yet calculated the gradient)
        self.current_state = "INIT"

        self.gradient = []

    def __call__(self, rho_vector=None, need_value=True, need_gradient=True):
        """Evaluate value and/or gradient of objective function.
        """
        if rho_vector:
            self.update_design(rho_vector=rho_vector)

        # Run forward run if requested
        if need_value and self.current_state == "INIT":
            print("Starting forward run...")
            self.forward_run()

        # Run adjoint simulation and calculate gradient if requested
        if need_gradient:
            if self.current_state == "INIT":
                # we need to run a forward run before an adjoint run
                print("Starting forward run...")
                self.forward_run()
                print("Starting adjoint run...")
                self.a_E = []
                self.adjoint_run()
                print("Calculating gradient...")
                self.calculate_gradient()
            elif self.current_state == "FWD":
                print("Starting adjoint run...")
                self.a_E = []
                self.adjoint_run()
                print("Calculating gradient...")
                self.calculate_gradient()
            else:
                raise ValueError("Incorrect solver state detected: {}".format(
                    self.current_state))

        return self.f0, self.gradient

    def get_fdf_funcs(self):
        """construct callable functions for objective function value and gradient

        Returns
        -------
        2-tuple (f_func, df_func) of standalone (non-class-method) callables, where
            f_func(beta) = objective function value for design variables beta
           df_func(beta) = objective function gradient for design variables beta
        """
        def _f(x=None):
            (fq, _) = self.__call__(rho_vector=x, need_gradient=False)
            return fq

        def _df(x=None):
            (_, df) = self.__call__(need_value=False)
            return df

        return _f, _df

    def prepare_forward_run(self):
        # prepare forward run
        self.sim.reset_meep()

        # add forward sources
        self.sim.change_sources(self.forward_sources)

        # register user specified monitors
        self.forward_monitors = [
        ]  # save reference to monitors so we can track dft convergence
        for m in self.objective_arguments:
            self.forward_monitors.append(m.register_monitors(self.frequencies))

        # register design region
        self.design_region_monitors = [
            self.sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez],
                                    self.frequencies,
                                    where=dr.volume,
                                    yee_grid=True)
            for dr in self.design_regions
        ]

        # store design region voxel parameters
        self.design_grids = []
        for drm in self.design_region_monitors:
            s = [
                self.sim.get_array_slice_dimensions(c, vol=drm.where)[0]
                for c in [mp.Ex, mp.Ey, mp.Ez]
            ]
            self.design_grids += [YeeDims(*s)]

    def forward_run(self):
        # set up monitors
        self.prepare_forward_run()

        # Forward run
        self.sim.run(until_after_sources=stop_when_dft_decayed(
            self.sim, self.design_region_monitors, self.decay_dt,
            self.decay_fields, self.fcen_idx, self.decay_by, True,
            self.minimum_run_time, self.maximum_run_time))

        # record objective quantities from user specified monitors
        self.results_list = []
        for m in self.objective_arguments:
            self.results_list.append(m())

        # evaluate objectives
        self.f0 = [fi(*self.results_list) for fi in self.objective_functions]
        if len(self.f0) == 1:
            self.f0 = self.f0[0]

        # Store forward fields for each set of design variables in array (x,y,z,field_components,frequencies)
        self.d_E = [[
            np.zeros((self.nf, c[0], c[1], c[2]), dtype=np.complex128)
            for c in dg
        ] for dg in self.design_grids]
        for nb, dgm in enumerate(self.design_region_monitors):
            for ic, c in enumerate([mp.Ex, mp.Ey, mp.Ez]):
                for f in range(self.nf):
                    self.d_E[nb][ic][f, :, :, :] = atleast_3d(
                        self.sim.get_dft_array(dgm, c, f))

        # store objective function evaluation in memory
        self.f_bank.append(self.f0)

        # update solver's current state
        self.current_state = "FWD"

    def prepare_adjoint_run(self):
        # Compute adjoint sources
        self.adjoint_sources = [[]
                                for i in range(len(self.objective_functions))]
        for ar in range(len(self.objective_functions)):
            for mi, m in enumerate(self.objective_arguments):
                dJ = jacobian(self.objective_functions[ar],
                              mi)(*self.results_list
                                  )  # get gradient of objective w.r.t. monitor
                if np.any(dJ):
                    self.adjoint_sources[ar] += m.place_adjoint_source(
                        dJ)  # place the appropriate adjoint sources

    def adjoint_run(self):
        # set up adjoint sources and monitors
        self.prepare_adjoint_run()
        for ar in range(len(self.objective_functions)):
            # Reset the fields
            self.sim.reset_meep()

            # Update the sources
            self.sim.change_sources(self.adjoint_sources[ar])

            # register design flux
            self.design_region_monitors = [
                self.sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez],
                                        self.frequencies,
                                        where=dr.volume,
                                        yee_grid=True)
                for dr in self.design_regions
            ]

            # Adjoint run
            self.sim.run(until_after_sources=stop_when_dft_decayed(
                self.sim, self.design_region_monitors, self.decay_dt,
                self.decay_fields, self.fcen_idx, self.decay_by, True,
                self.minimum_run_time, self.maximum_run_time))

            # Store adjoint fields for each design set of design variables in array (x,y,z,field_components,frequencies)
            self.a_E.append([[
                np.zeros((self.nf, c[0], c[1], c[2]), dtype=np.complex128)
                for c in dg
            ] for dg in self.design_grids])
            for nb, dgm in enumerate(self.design_region_monitors):
                for ic, c in enumerate([mp.Ex, mp.Ey, mp.Ez]):
                    for f in range(self.nf):
                        self.a_E[ar][nb][ic][f, :, :, :] = atleast_3d(
                            self.sim.get_dft_array(dgm, c, f))

        # update optimizer's state
        self.current_state = "ADJ"

    def calculate_gradient(self):
        # Iterate through all design regions and calculate gradient
        self.gradient = [[
            dr.get_gradient(self.sim, self.a_E[ar][dri], self.d_E[dri],
                            self.frequencies)
            for dri, dr in enumerate(self.design_regions)
        ] for ar in range(len(self.objective_functions))]

        # Cleanup list of lists
        if len(self.gradient) == 1:
            self.gradient = self.gradient[0]  # only one objective function
            if len(self.gradient) == 1:
                self.gradient = self.gradient[
                    0]  # only one objective function and one design region
        else:
            if len(self.gradient[0]) == 1:
                self.gradient = [
                    g[0] for g in self.gradient
                ]  # multiple objective functions bu one design region
        # Return optimizer's state to initialization
        self.current_state = "INIT"

    def calculate_fd_gradient(self,
                              num_gradients=1,
                              db=1e-4,
                              design_variables_idx=0,
                              filter=None):
        '''
        Estimate central difference gradients.

        Parameters
        ----------
        num_gradients ... : scalar
            number of gradients to estimate. Randomly sampled from parameters.
        db ... : scalar
            finite difference step size
        design_variables_idx ... : scalar
            which design region to pull design variables from

        Returns
        -----------
        fd_gradient ... : lists
            [number of objective functions][number of gradients]

        '''
        if filter is None:
            filter = lambda x: x
        if num_gradients > self.num_design_params[design_variables_idx]:
            raise ValueError(
                "The requested number of gradients must be less than or equal to the total number of design parameters."
            )

        # cleanup simulation object
        self.sim.reset_meep()
        self.sim.change_sources(self.forward_sources)

        # preallocate result vector
        fd_gradient = []

        # randomly choose indices to loop estimate
        fd_gradient_idx = np.random.choice(
            self.num_design_params[design_variables_idx],
            num_gradients,
            replace=False)

        for k in fd_gradient_idx:

            b0 = np.ones((self.num_design_params[design_variables_idx], ))
            b0[:] = (self.design_regions[design_variables_idx].
                     design_parameters.weights)
            # -------------------------------------------- #
            # left function evaluation
            # -------------------------------------------- #
            self.sim.reset_meep()

            # assign new design vector
            b0[k] -= db
            self.design_regions[design_variables_idx].update_design_parameters(
                b0)

            # initialize design monitors
            self.forward_monitors = []
            for m in self.objective_arguments:
                self.forward_monitors.append(
                    m.register_monitors(self.frequencies))

            self.sim.run(until_after_sources=stop_when_dft_decayed(
                self.sim, self.forward_monitors, self.decay_dt,
                self.decay_fields, self.fcen_idx, self.decay_by, True,
                self.minimum_run_time, self.maximum_run_time))

            # record final objective function value
            results_list = []
            for m in self.objective_arguments:
                results_list.append(m())
            fm = [fi(*results_list) for fi in self.objective_functions]

            # -------------------------------------------- #
            # right function evaluation
            # -------------------------------------------- #
            self.sim.reset_meep()

            # assign new design vector
            b0[k] += 2 * db  # central difference rule...
            self.design_regions[design_variables_idx].update_design_parameters(
                b0)

            # initialize design monitors
            self.forward_monitors = []
            for m in self.objective_arguments:
                self.forward_monitors.append(
                    m.register_monitors(self.frequencies))

            # add monitor used to track dft convergence
            self.sim.run(until_after_sources=stop_when_dft_decayed(
                self.sim, self.forward_monitors, self.decay_dt,
                self.decay_fields, self.fcen_idx, self.decay_by, True,
                self.minimum_run_time, self.maximum_run_time))

            # record final objective function value
            results_list = []
            for m in self.objective_arguments:
                results_list.append(m())
            fp = [fi(*results_list) for fi in self.objective_functions]

            # -------------------------------------------- #
            # estimate derivative
            # -------------------------------------------- #
            fd_gradient.append([
                np.squeeze((fp[fi] - fm[fi]) / (2 * db))
                for fi in range(len(self.objective_functions))
            ])

        # Cleanup singleton dimensions
        if len(fd_gradient) == 1:
            fd_gradient = fd_gradient[0]

        return fd_gradient, fd_gradient_idx

    def update_design(self, rho_vector):
        """Update the design permittivity function.

        rho_vector ....... a list of numpy arrays that maps to each design region
        """
        for bi, b in enumerate(self.design_regions):
            if np.array(rho_vector[bi]).ndim > 1:
                raise ValueError(
                    "Each vector of design variables must contain only one dimension."
                )
            b.update_design_parameters(rho_vector[bi])

        self.sim.reset_meep()
        self.current_state = "INIT"

    def get_objective_arguments(self):
        '''Return list of evaluated objective arguments.
        '''
        objective_args_evaluation = [
            m.get_evaluation() for m in self.objective_arguments
        ]
        return objective_args_evaluation

    def plot2D(self, init_opt=False, **kwargs):
        """Produce a graphical visualization of the geometry and/or fields,
           as appropriately autodetermined based on the current state of
           progress.
        """

        if init_opt:
            self.prepare_forward_run()

        self.sim.plot2D(**kwargs)


def stop_when_dft_decayed(simob,
                          mon,
                          dt,
                          c,
                          fcen_idx,
                          decay_by,
                          yee_grid=False,
                          minimum_run_time=0,
                          maximum_run_time=None):
    '''Step function that monitors the relative change in DFT fields for a list of monitors.

    mon ............. a list of monitors
    c ............... a list of components to monitor

    '''
    # get monitor dft output array dimensions
    dims = []
    for m in mon:
        ci_dims = []
        for ci in c:
            comp = ci if yee_grid else mp.Dielectric
            ci_dims += [simob.get_array_slice_dimensions(comp, vol=m.where)[0]]
        dims.append(ci_dims)

    # Record data in closure so that we can persitently edit
    closure = {'previous_fields': [[None for di in d] for d in dims], 't0': 0}

    def _stop(sim):
        if sim.round_time() <= dt + closure['t0']:
            return False
        elif maximum_run_time and sim.round_time() > maximum_run_time:
            return True
        else:
            previous_fields = closure['previous_fields']

            # Pull all the relevant frequency and spatial dft points
            relative_change = []
            current_fields = [[0 for di in d] for d in dims]
            for mi, m in enumerate(mon):
                for ic, cc in enumerate(c):
                    if isinstance(m, mp.DftFlux):
                        current_fields[mi][ic] = mp.get_fluxes(m)[fcen_idx]
                    elif isinstance(m, mp.DftFields):
                        current_fields[mi][ic] = atleast_3d(
                            sim.get_dft_array(m, cc, fcen_idx))
                    elif isinstance(m, mp.simulation.DftNear2Far):
                        current_fields[mi][ic] = atleast_3d(
                            sim.get_dft_array(m, cc, fcen_idx))
                    else:
                        raise TypeError(
                            "Monitor of type {} not supported".format(type(m)))

                    if previous_fields[mi][ic] is not None:
                        if np.sum(
                                np.abs(previous_fields[mi][ic] -
                                       current_fields[mi][ic])) == 0:
                            relative_change.append(0)
                        elif np.all(np.abs(previous_fields[mi][ic])):
                            relative_change_raw = np.abs(
                                previous_fields[mi][ic] -
                                current_fields[mi][ic]) / np.abs(
                                    previous_fields[mi][ic])
                            relative_change.append(
                                np.mean(relative_change_raw.flatten())
                            )  # average across space and frequency
                        else:
                            relative_change.append(1)
                    else:
                        relative_change.append(1)

            relative_change = np.mean(
                relative_change)  # average across monitors
            closure['previous_fields'] = current_fields
            closure['t0'] = sim.round_time()

            if mp.verbosity > 0:
                fmt = "DFT decay(t = {0:1.1f}): {1:0.4e}"
                print(fmt.format(sim.meep_time(), np.real(relative_change)))
            return relative_change <= decay_by and sim.round_time(
            ) >= minimum_run_time

    return _stop


def atleast_3d(*arys):
    from numpy import array, asanyarray, newaxis
    '''
    Modified version of numpy's `atleast_3d`

    Keeps one dimensional array data in first dimension, as
    opposed to moving it to the second dimension as numpy's
    version does. Keeps the meep dimensionality convention.

    View inputs as arrays with at least three dimensions.
    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted to
        arrays.  Arrays that already have three or more dimensions are
        preserved.
    Returns
    -------
    res1, res2, ... : ndarray
        An array, or list of arrays, each with ``a.ndim >= 3``.  Copies are
        avoided where possible, and views with three or more dimensions are
        returned.  For example, a 1-D array of shape ``(N,)`` becomes a view
        of shape ``(N, 1, 1)``, and a 2-D array of shape ``(M, N)`` becomes a
        view of shape ``(M, N, 1)``.
    '''
    res = []
    for ary in arys:
        ary = asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1, 1)
        elif ary.ndim == 1:
            result = ary[:, newaxis, newaxis]
        elif ary.ndim == 2:
            result = ary[:, :, newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res
