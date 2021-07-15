from typing import List, Iterable, Tuple

import meep as mp
import numpy as onp

from . import ObjectiveQuantity, DesignRegion

# Meep field components used to compute adjoint sensitivities
_ADJOINT_FIELD_COMPONENTS = [mp.Ex, mp.Ey, mp.Ez]

# The frequency axis in the array returned by `mp._get_gradient()`
_GRADIENT_FREQ_AXIS = 1


def _make_at_least_nd(x: onp.ndarray, dims: int = 3) -> onp.ndarray:
    """Makes an array have at least the specified number of dimensions."""
    return onp.reshape(x, x.shape + onp.maximum(dims - x.ndim, 0) * (1, ))


def calculate_vjps(
    simulation: mp.Simulation,
    design_regions: List[DesignRegion],
    frequencies: List[float],
    fwd_fields: List[List[onp.ndarray]],
    adj_fields: List[List[onp.ndarray]],
    design_variable_shapes: List[Tuple[int, ...]],
    sum_freq_partials: bool = True,
) -> List[onp.ndarray]:
    """Calculates the VJP for a given set of forward and adjoint fields."""
    vjps = [
        design_region.get_gradient(
            simulation,
            adj_fields[i],
            fwd_fields[i],
            frequencies,
        ) for i, design_region in enumerate(design_regions)
    ]
    if sum_freq_partials:
        vjps = [
            onp.sum(vjp, axis=_GRADIENT_FREQ_AXIS).reshape(shape)
            for vjp, shape in zip(vjps, design_variable_shapes)
        ]
    else:
        vjps = [
            vjp.reshape(shape + (-1, ))
            for vjp, shape in zip(vjps, design_variable_shapes)
        ]
    return vjps


def register_monitors(
    monitors: List[ObjectiveQuantity],
    frequencies: List[float],
) -> None:
    """Registers a list of monitors."""
    for monitor in monitors:
        monitor.register_monitors(frequencies)


def install_design_region_monitors(
    simulation: mp.Simulation,
    design_regions: List[DesignRegion],
    frequencies: List[float],
) -> List[mp.DftFields]:
    """Installs DFT field monitors at the design regions of the simulation."""
    design_region_monitors = [
        simulation.add_dft_fields(
            _ADJOINT_FIELD_COMPONENTS,
            frequencies,
            where=design_region.volume,
            yee_grid=True,
        ) for design_region in design_regions
    ]
    return design_region_monitors


def gather_monitor_values(monitors: List[ObjectiveQuantity]) -> onp.ndarray:
    """Gathers the mode monitor overlap values as a rank 2 ndarray.

    Args:
      monitors: the mode monitors.

  Returns:
    a rank-2 ndarray, where the dimensions are (monitor, frequency), of dtype
    complex128.  Note that these values refer to the mode as oriented (i.e. they
    are unidirectional).
  """
    monitor_values = []
    for monitor in monitors:
        monitor_values.append(monitor())
    monitor_values = onp.array(monitor_values)
    assert monitor_values.ndim in [1, 2]
    monitor_values = _make_at_least_nd(monitor_values, 2)
    return monitor_values


def gather_design_region_fields(
    simulation: mp.Simulation,
    design_region_monitors: List[mp.DftFields],
    frequencies: List[float],
) -> List[List[onp.ndarray]]:
    """Collects the design region DFT fields from the simulation.

  Args:
   simulation: the simulation object.
   design_region_monitors: the installed design region monitors.
   frequencies: the frequencies to monitor.

  Returns:
    A list of lists.  Each entry (list) in the overall list corresponds one-to-
    one with a declared design region.  For each such contained list, the
    entries correspond to the field components that are monitored.  The entries
    are ndarrays of rank 4 with dimensions (freq, x, y, (z-or-pad)).

    The design region fields are sampled on the *Yee grid*.  This makes them
    fairly awkward to inspect directly.  Their primary use case is supporting
    gradient calculations.
  """
    fwd_fields = []
    for monitor in design_region_monitors:
        fields_by_component = []
        for component in _ADJOINT_FIELD_COMPONENTS:
            fields_by_freq = []
            for freq_idx, _ in enumerate(frequencies):
                fields = simulation.get_dft_array(monitor, component, freq_idx)
                fields_by_freq.append(_make_at_least_nd(fields))
            fields_by_component.append(onp.stack(fields_by_freq))
        fwd_fields.append(fields_by_component)
    return fwd_fields


def validate_and_update_design(
        design_regions: List[DesignRegion],
        design_variables: Iterable[onp.ndarray]) -> None:
    """Validate the design regions and variables.

  In particular the design variable should be 1,2,3-D and the design region
  shape should match the design variable shape after dimension expansion.
  The arguments are modified in place.

    Args:
      design_regions: List of mpa.DesignRegion,
      design_variables: Iterable with numpy arrays representing design variables.

  Raises:
    ValueError if the validation of dimensions fails.
  """
    for i, (design_region,
            design_variable) in enumerate(zip(design_regions,
                                              design_variables)):
        if design_variable.ndim not in [1, 2, 3]:
            raise ValueError(
                'Design variables should be 1D, 2D, or 3D, but the design variable '
                'at index {} had a shape of {}.'.format(
                    i, design_variable.shape))
        design_region_shape = tuple(
            int(x) for x in design_region.design_parameters.grid_size)
        design_variable_padded_shape = design_variable.shape + (1, ) * (
            3 - design_variable.ndim)
        if design_variable_padded_shape != design_region_shape:
            raise ValueError(
                'The design variable at index {} with a shape of {} is '
                'incompatible with the associated design region, which has a shape '
                'of {}.'.format(i, design_variable.shape, design_region_shape))
        design_variable = onp.asarray(design_variable, dtype=onp.float64)
        # Update the design variable in Meep
        design_region.update_design_parameters(design_variable.flatten())


def create_adjoint_sources(
        monitors: Iterable[ObjectiveQuantity],
        monitor_values_grad: onp.ndarray) -> List[mp.Source]:
    monitor_values_grad = onp.asarray(monitor_values_grad,
                                      dtype=onp.complex128)
    if not onp.any(monitor_values_grad):
        raise RuntimeError(
            'The gradient of all monitor values is zero, which '
            'means that no adjoint sources can be placed to set '
            'up an adjoint simulation in Meep. Possible causes '
            'could be:\n\n'
            ' * the forward simulation was not run for long enough '
            'to allow the input pulse(s) to reach the monitors'
            ' * the monitor values are disconnected from the '
            'objective function output.')
    adjoint_sources = []
    for monitor_idx, monitor in enumerate(monitors):
        # `dj` for each monitor will have a shape of (num frequencies,)
        dj = onp.asarray(monitor_values_grad[monitor_idx],
                         dtype=onp.complex128)
        if onp.any(dj):
            adjoint_sources += monitor.place_adjoint_source(dj)
    assert adjoint_sources
    return adjoint_sources
