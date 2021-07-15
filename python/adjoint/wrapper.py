"""Wrapper for converting a Meep simulation into a differentiable JAX callable function.

Usage example:
```
import jax.numpy as jnp
import meep as mp
import meep.adjoint as mpa

sources = [
  mp.EigenModeSource(...)
]

monitors = [
  mpa.EigenmodeCoefficient(...),
  mpa.EigenmodeCoefficient(...),
]

design_regions = [
  mpa.DesignRegion(...)
]

frequencies = [1/1.55, 1/1.60, 1/1.65, ...]

simulation = mp.Simulation(...)

wrapped_meep = MeepJaxWrapper(
    simulation,
    sources,
    monitors,
    design_regions,
    frequencies,
    measurement_interval = 50.0,
    dft_field_components = (mp.Ez,),
    dft_threshold = 1e-6,
    minimum_run_time = 0,
    maximum_run_time = onp.inf,
    until_after_sources = True
)

def loss(x):
    monitor_values = wrapped_meep([x])
    t = monitor_values[0,:] / monitor_values[1,:]
    # Mean transmission vs wavelength
    return jnp.mean(jnp.abs(t))

value, grad = jax.value_and_grad(loss)(x)
```
"""

from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import meep as mp
import numpy as onp

from . import utils
from . import DesignRegion, EigenmodeCoefficient

_norm_fn = onp.linalg.norm
_reduce_fn = onp.max


class MeepJaxWrapper:
    """Wraps a Meep simulation object into a JAX-differentiable callable.

  Attributes:
      simulation: the pre-configured Meep simulation object.
      sources: a list of Meep sources for the forward simulation.
      monitors: a list of eigenmode coefficient monitors from the `meep.adjoint`
        module.
      design_regions: a list of design regions from the `meep.adjoint` module.
      frequencies: the list of frequencies, in normalized Meep units.
      measurement_interval: the time interval between DFT field convergence
        measurements, in Meep time units. The default value is 50.
      dft_field_components: a list of Meep field components, such as `mp.Ex`,
        `mp.Hy`, etc, whose DFT will be monitored for convergence to stop the
        simulation. The default is `mp.Ez`.
      dft_threshold: the threshold for DFT field convergence. Once the norm of the
        change in the fields (the maximum over all design regions and field
        components) is less than this value, the simulation will be stopped. The
        default value is 1e-6.
      minimum_run_time: the minimum run time of the simulation, in Meep time
        units. The default value is 0.
      maximum_run_time: the maximum run time of the simulation, in Meep time
        units. The default value is infinity.
      until_after_sources: whether `maximum_run_time` should be ignored until the
        sources have turned off. This parameter specifies whether `until` or
        `until_after_sources` is used. See
        https://meep.readthedocs.io/en/latest/Python_User_Interface/#Simulation
          for more information. The default is true.
  """
    _log_fn = print

    def __init__(self,
                 simulation: mp.Simulation,
                 sources: List[mp.Source],
                 monitors: List[EigenmodeCoefficient],
                 design_regions: List[DesignRegion],
                 frequencies: List[float],
                 measurement_interval: float = 50.0,
                 dft_field_components: Tuple[int, ...] = (mp.Ez, ),
                 dft_threshold: float = 1e-6,
                 minimum_run_time: float = 0,
                 maximum_run_time: float = onp.inf,
                 until_after_sources: bool = True):
        self.simulation = simulation
        self.sources = sources
        self.monitors = monitors
        self.design_regions = design_regions
        self.frequencies = frequencies
        self.measurement_interval = measurement_interval
        self.dft_field_components = dft_field_components
        self.dft_threshold = dft_threshold
        self.minimum_run_time = minimum_run_time
        self.maximum_run_time = maximum_run_time
        self.until_after_sources = until_after_sources

        self._reset_convergence_measurement()
        self._simulate_fn = self._initialize_callable()

    def __call__(self, designs: List[jnp.ndarray]) -> jnp.ndarray:
        """Performs a Meep simulation, taking a list of designs and returning mode overlaps.

    Args:
      designs: a list of design variables as 1D, 2D, or 3D JAX arrays. Valid shapes for
      design variables are (Nx, Ny, Nz) where Nx{y,z} match the elements of the
      `grid_size` constructor argument of Meep's `MaterialGrid` used for the
      corresponding design region. Singleton dimensions of the `grid_size` may be
      omitted from the corresponding design variable. For example, a design variable
      with a shape of either (10, 20) or (10, 20, 1) would be compatible with a
      `grid_size` of (10, 20, 1). Similarly, a design variable with shapes of (25,),
      (25, 1), or (25, 1, 1) would be compatible with a `grid_size` of (25, 1, 1).

    Returns:
      a complex-valued JAX ndarray of differentiable mode monitor overlaps values with
      a shape of (num monitors, num frequencies).
    """
        return self._simulate_fn(designs)

    def _reset_convergence_measurement(self,
                                       monitors: List[mp.DftFields] = None
                                       ) -> None:
        """Resets the DFT convergence measurement."""
        if monitors is None:
            monitors = []
        self._dft_convergence_monitors = monitors
        self._last_measurement_meep_time = 0.0
        self._previous_fields = self._init_empty_dft_field_container()
        self._dft_relative_change = []

    def _run_fwd_simulation(self, design_variables):
        """Runs forward simulation, returning monitor values and design region fields."""
        utils.validate_and_update_design(self.design_regions, design_variables)
        self.simulation.reset_meep()
        self.simulation.change_sources(self.sources)
        utils.register_monitors(self.monitors, self.frequencies)
        design_region_monitors = utils.install_design_region_monitors(
            self.simulation,
            self.design_regions,
            self.frequencies,
        )
        self.simulation.init_sim()
        sim_run_args = {
            'until_after_sources' if self.until_after_sources else 'until':
            self._simulation_run_callback
        }
        self._reset_convergence_measurement(design_region_monitors)
        self.simulation.run(**sim_run_args)

        monitor_values = utils.gather_monitor_values(self.monitors)
        fwd_fields = utils.gather_design_region_fields(
            self.simulation,
            design_region_monitors,
            self.frequencies,
        )
        return (jnp.asarray(monitor_values),
                jax.tree_map(jnp.asarray, fwd_fields))

    def _run_adjoint_simulation(self, monitor_values_grad):
        """Runs adjoint simulation, returning design region fields."""
        if not self.design_regions:
            raise RuntimeError(
                'An adjoint simulation was attempted when no design '
                'regions are present.')
        adjoint_sources = utils.create_adjoint_sources(self.monitors,
                                                       monitor_values_grad)
        self.simulation.reset_meep()
        self.simulation.change_sources(adjoint_sources)
        design_region_monitors = utils.install_design_region_monitors(
            self.simulation,
            self.design_regions,
            self.frequencies,
        )
        self.simulation.init_sim()
        sim_run_args = {
            'until_after_sources' if self.until_after_sources else 'until':
            self._simulation_run_callback
        }
        self._reset_convergence_measurement(design_region_monitors)
        self.simulation.run(**sim_run_args)

        return utils.gather_design_region_fields(self.simulation,
                                                 design_region_monitors,
                                                 self.frequencies)

    def _calculate_vjps(self,
                        fwd_fields,
                        adj_fields,
                        design_variable_shapes,
                        sum_freq_partials=True):
        """Calculates the VJP for a given set of forward and adjoint fields."""
        return utils.calculate_vjps(
            self.simulation,
            self.design_regions,
            self.frequencies,
            fwd_fields,
            adj_fields,
            design_variable_shapes,
            sum_freq_partials=sum_freq_partials,
        )

    def _initialize_callable(
            self) -> Callable[[List[jnp.ndarray]], jnp.ndarray]:
        """Initializes the callable JAX function and registers its VJP."""
        @jax.custom_vjp
        def simulate(design_variables: List[jnp.ndarray]) -> jnp.ndarray:
            monitor_values, _ = self._run_fwd_simulation(design_variables)
            return monitor_values

        def _simulate_fwd(design_variables):
            """Runs forward simulation, returning monitor values and fields."""
            monitor_values, fwd_fields = self._run_fwd_simulation(
                design_variables)
            design_variable_shapes = [x.shape for x in design_variables]
            return monitor_values, (fwd_fields, design_variable_shapes)

        def _simulate_rev(res, monitor_values_grad):
            """Runs adjoint simulation, returning VJP of design wrt monitor values."""
            fwd_fields = jax.tree_map(
                lambda x: onp.asarray(x, dtype=onp.complex128), res[0])
            design_variable_shapes = res[1]
            adj_fields = self._run_adjoint_simulation(monitor_values_grad)
            vjps = self._calculate_vjps(fwd_fields, adj_fields,
                                        design_variable_shapes)
            return ([jnp.asarray(vjp) for vjp in vjps], )

        simulate.defvjp(_simulate_fwd, _simulate_rev)

        return simulate

    def _init_empty_dft_field_container(self) -> List[List[float]]:
        """Initializes a nested list for storing DFT fields for convergence monitoring."""
        num_components = len(self.dft_field_components)
        num_monitors = len(self._dft_convergence_monitors)
        return [[0.0 for _ in range(num_components)]
                for _ in range(num_monitors)]

    def _are_dfts_converged(self, sim: mp.Simulation) -> bool:
        """Callback to determine whether the DFT fields are converged below the threshold."""
        if self.dft_threshold == 0 or not self._dft_convergence_monitors or not self.dft_field_components:
            return False
        relative_change = []
        current_fields = self._init_empty_dft_field_container()
        for monitor_idx, monitor in enumerate(self._dft_convergence_monitors):
            for component_idx, component in enumerate(
                    self.dft_field_components):
                previous_fields = self._previous_fields[monitor_idx][
                    component_idx]
                current_fields[monitor_idx][component_idx] = sim.get_dft_array(
                    monitor,
                    component,
                    int(monitor.nfreqs // 2),
                )
                norm_previous = _norm_fn(previous_fields)
                field_diff = previous_fields - current_fields[monitor_idx][
                    component_idx]
                if norm_previous != 0:
                    relative_change.append(
                        _norm_fn(field_diff) / norm_previous)
                else:
                    relative_change.append(1.0)
        relative_change = _reduce_fn(relative_change)
        self._dft_relative_change.append(relative_change)
        self._previous_fields = current_fields
        if mp.am_master() and mp.verbosity > 0:
            print(
                'At simulation time %.2f the relative change in the DFT fields is %.2e.'
                % (sim.meep_time(), relative_change))
        return relative_change < self.dft_threshold

    def _simulation_run_callback(self, sim: mp.Simulation) -> bool:
        """A callback function returning `True` when the simulation should stop.

    This is a step function that gets called at each time step of the Meep
    simulation, taking a Meep simulation object as an input and returning `True`
    when the simulation should be terminated and returning `False` when the
    simulation should continue. The resulting step function is designed to be
    used with the `until` and `until_after_sources` arguments of the Meep
    simulation `run()` routine.

    Args:
      sim: a Meep simulation object.

    Returns:
      a boolean indicating whether the simulation should be terminated.
    """
        current_meep_time = sim.round_time()
        if current_meep_time <= self._last_measurement_meep_time + self.measurement_interval:
            return False
        if current_meep_time <= self.minimum_run_time:
            if mp.am_master() and mp.verbosity > 0:
                remaining_time = self.minimum_run_time - sim.round_time()
                self._log_fn(
                    '%.2f to go until the minimum simulation runtime is reached.'
                    % (remaining_time, ))
            return False
        if current_meep_time >= self.maximum_run_time:
            if mp.am_master() and mp.verbosity > 0:
                self._log_fn(
                    'Stopping the simulation because the maximum simulation run '
                    'time of %.2f has been reached.' %
                    (self.maximum_run_time, ))
            return True
        self._last_measurement_meep_time = current_meep_time
        return self._are_dfts_converged(sim)
