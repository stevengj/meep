import unittest
import parameterized

from utils import VectorComparisonMixin

import jax
import jax.numpy as jnp
import meep as mp
import meep.adjoint as mpa
import numpy as onp

# The calculation of finite difference gradients requires that JAX be operated with double precision
jax.config.update('jax_enable_x64', True)

# The step size for the finite difference gradient calculation
_FD_STEP = 1e-4

# The tolerance for the adjoint and finite difference gradient comparison
_TOL = 2e-2

mp.verbosity(0)


def build_straight_wg_simulation(
  wg_width=0.5,
  wg_padding=1.0,
  wg_length=1.0,
  pml_width=1.0,
  source_to_pml=0.5,
  source_to_monitor=0.1,
  frequencies=[1 / 1.55],
  gaussian_rel_width=0.2,
  sim_resolution=20,
  design_region_resolution=20,
):
  """Builds a simulation of a straight waveguide with a design region segment."""
  design_region_shape = (1.0, wg_width)

  # Simulation domain size
  sx = 2 * pml_width + 2 * wg_length + design_region_shape[0]
  sy = 2 * pml_width + 2 * wg_padding + max(
    wg_width,
    design_region_shape[1],
  )

  # Mean / center frequency
  fmean = onp.mean(frequencies)

  si = mp.Medium(index=3.4)
  sio2 = mp.Medium(index=1.44)

  sources = [
    mp.EigenModeSource(
      mp.GaussianSource(frequency=fmean, fwidth=fmean * gaussian_rel_width),
      eig_band=1,
      direction=mp.NO_DIRECTION,
      eig_kpoint=mp.Vector3(1, 0, 0),
      size=mp.Vector3(0, wg_width + 2 * wg_padding, 0),
      center=[-sx / 2 + pml_width + source_to_pml, 0, 0],
    ),
    mp.EigenModeSource(
      mp.GaussianSource(frequency=fmean, fwidth=fmean * gaussian_rel_width),
      eig_band=1,
      direction=mp.NO_DIRECTION,
      eig_kpoint=mp.Vector3(-1, 0, 0),
      size=mp.Vector3(0, wg_width + 2 * wg_padding, 0),
      center=[sx / 2 - pml_width - source_to_pml, 0, 0],
    ),
  ]

  nx = int(design_region_resolution * design_region_shape[0])
  ny = int(design_region_resolution * design_region_shape[1])
  mat_grid = mp.MaterialGrid(
    mp.Vector3(nx, ny),
    sio2,
    si,
    grid_type='U_DEFAULT',
  )

  design_regions = [
    mpa.DesignRegion(
      mat_grid,
      volume=mp.Volume(
        center=mp.Vector3(),
        size=mp.Vector3(
          design_region_shape[0],
          design_region_shape[1],
          0,
        ),
      ),
    )
  ]

  geometry = [
    mp.Block(
      center=mp.Vector3(x=-design_region_shape[0] / 2 - wg_length / 2 - pml_width / 2),
      material=si,
      size=mp.Vector3(wg_length + pml_width, wg_width, 0)),  # left wg
    mp.Block(
      center=mp.Vector3(x=+design_region_shape[0] / 2 + wg_length / 2 + pml_width / 2),
      material=si,
      size=mp.Vector3(wg_length + pml_width, wg_width, 0)),  # right wg
    mp.Block(
      center=design_regions[0].center,
      size=design_regions[0].size,
      material=mat_grid),  # design region
  ]

  simulation = mp.Simulation(
    cell_size=mp.Vector3(sx, sy),
    boundary_layers=[mp.PML(pml_width)],
    geometry=geometry,
    sources=sources,
    resolution=sim_resolution,
  )

  monitor_centers = [
    mp.Vector3(-sx / 2 + pml_width + source_to_pml + source_to_monitor),
    mp.Vector3(sx / 2 - pml_width - source_to_pml - source_to_monitor),
  ]
  monitor_size = mp.Vector3(y=wg_width + 2 * wg_padding)

  monitors = [
    mpa.EigenmodeCoefficient(
      simulation,
      mp.Volume(center=center, size=monitor_size),
      mode=1,
      forward=forward) for center in monitor_centers for forward in [True, False]
  ]
  return simulation, sources, monitors, design_regions, frequencies


class UtilsTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    (
      self.simulation,
      self.sources,
      self.monitors,
      self.design_regions,
      self.frequencies,
    ) = build_straight_wg_simulation()

  def test_mode_monitor_helpers(self):
    mpa.utils.register_monitors(self.monitors, self.frequencies)
    self.simulation.run(until=100)
    monitor_values = mpa.utils.gather_monitor_values(self.monitors)
    self.assertEqual(monitor_values.dtype, onp.complex128)
    self.assertEqual(monitor_values.shape,
                     (len(self.monitors), len(self.frequencies)))

  def test_design_region_monitor_helpers(self):
    design_region_monitors = mpa.utils.install_design_region_monitors(
      self.simulation,
      self.design_regions,
      self.frequencies,
    )
    self.simulation.run(until=100)
    design_region_fields = mpa.utils.gather_design_region_fields(
      self.simulation,
      design_region_monitors,
      self.frequencies,
    )

    self.assertIsInstance(design_region_fields, list)
    self.assertEqual(len(design_region_fields), len(self.design_regions))

    self.assertIsInstance(design_region_fields[0], list)
    self.assertEqual(len(design_region_fields[0]), len(mpa.utils._ADJOINT_FIELD_COMPONENTS))

    for value in design_region_fields[0]:
      self.assertIsInstance(value, onp.ndarray)
      self.assertEqual(value.ndim, 4)  # dims: freq, x, y, pad
      self.assertEqual(value.dtype, onp.complex128)


class WrapperTest(VectorComparisonMixin, unittest.TestCase):

  @parameterized.parameterized.expand([
    ('1500_1550bw_01relative_gaussian_port1', onp.linspace(1 / 1.50, 1 / 1.55, 3).tolist(), 0.1, 1.0, 0),
    # ('1550_1600bw_02relative_gaussian_port1', onp.linspace(1 / 1.55, 1 / 1.60, 3).tolist(), 0.2, 1.0, 0),
    # ('1500_1600bw_03relative_gaussian_port1', onp.linspace(1 / 1.50, 1 / 1.60, 4).tolist(), 0.3, 1.0, 0),
    ('1500_1550bw_01relative_gaussian_port2', onp.linspace(1 / 1.50, 1 / 1.55, 3).tolist(), 0.1, 1.0, 1),
    # ('1550_1600bw_02relative_gaussian_port2', onp.linspace(1 / 1.55, 1 / 1.60, 3).tolist(), 0.2, 1.0, 1),
    # ('1500_1600bw_03relative_gaussian_port2', onp.linspace(1 / 1.50, 1 / 1.60, 4).tolist(), 0.3, 1.0, 1),
  ])
  def test_wrapper_gradients(self, _, frequencies, gaussian_rel_width, design_variable_fill_value, excite_port_idx):
    """Tests gradient from the JAX-Meep wrapper against finite differences."""
    (
      simulation,
      sources,
      monitors,
      design_regions,
      frequencies,
    ) = build_straight_wg_simulation(frequencies=frequencies, gaussian_rel_width=gaussian_rel_width)

    design_shape = tuple(int(i) for i in design_regions[0].design_parameters.grid_size)[:2]
    x = onp.ones(design_shape) * design_variable_fill_value

    # Define a loss function
    def loss_fn(x, excite_port_idx=0):
      wrapped_meep = mpa.MeepJaxWrapper(
        simulation,
        [sources[excite_port_idx]],
        monitors,
        design_regions,
        frequencies,
      )
      monitor_values = wrapped_meep([x])
      s1p, s1m, s2m, s2p = monitor_values
      if excite_port_idx == 0:
        t = s2m / s1p
      else:
        t = s1m / s2p
      # Mean transmission vs wavelength
      t_mean = jnp.mean(jnp.square(jnp.abs(t)))
      return t_mean

    value, adjoint_grad = jax.value_and_grad(loss_fn)(x, excite_port_idx=excite_port_idx)

    projection = []
    fd_projection = []

    # Project along 5 random directions in the design parameter space.
    for seed in range(5):
      # Create dp
      random_perturbation_vector = _FD_STEP * jax.random.normal(
        jax.random.PRNGKey(seed),
        x.shape,
      )

      # Calculate p + dp
      x_perturbed = x + random_perturbation_vector

      # Calculate T(p + dp)
      value_perturbed = loss_fn(x_perturbed, excite_port_idx=excite_port_idx)

      projection.append(
        onp.dot(
          random_perturbation_vector.ravel(),
          adjoint_grad.ravel(),
        ))
      fd_projection.append(value_perturbed - value)

    projection = onp.stack(projection)
    fd_projection = onp.stack(fd_projection)

    # Check that dp . ∇T ~ T(p + dp) - T(p)
    self.assertVectorsClose(
      projection,
      fd_projection,
      epsilon=_TOL,
    )


if __name__ == '__main__':
  unittest.main()
