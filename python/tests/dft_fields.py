import unittest
import h5py
import numpy as np
import meep as mp
import tempfile
import os

class TestDFTFields(unittest.TestCase):

    def init(self):
        resolution = 10
        n = 3.4
        w = 1.0
        r = 1.0
        pad = 4
        self.dpml = 2
        self.sxy = 2.0 * (r + w + pad + self.dpml)
        cell = mp.Vector3(self.sxy, self.sxy)
        pml_layers = [mp.PML(self.dpml)]

        geometry = [
            mp.Cylinder(r + w, material=mp.Medium(epsilon=n * n)),
            mp.Cylinder(r, material=mp.vacuum),
        ]
        self.fcen = 0.118
        self.df = 0.1

        src = mp.GaussianSource(self.fcen, fwidth=self.df)
        sources = [mp.Source(src=src, component=mp.Ez, center=mp.Vector3(r + 0.1))]

        return mp.Simulation(
            cell_size=cell,
            resolution=resolution,
            geometry=geometry,
            sources=sources,
            boundary_layers=pml_layers,
        )
    def test_use_centered_grid(self):
        sim = self.init()
        sim.init_sim()
        dft_fields = sim.add_dft_fields([mp.Ez], self.fcen, self.fcen, 1, yee_grid=True)
        sim.run(until=100)
        
    def test_get_dft_array(self):
        sim = self.init()
        sim.init_sim()
        dft_fields = sim.add_dft_fields([mp.Ez], self.fcen, self.fcen, 1)
        fr = mp.FluxRegion(mp.Vector3(), size=mp.Vector3(self.sxy, self.sxy), direction=mp.X)
        dft_flux = sim.add_flux(self.fcen, 0, 1, fr)

        # volumes with zero thickness in x and y directions to test collapsing
        # of empty dimensions in DFT array and HDF5 output routines
        thin_x_volume = mp.Volume(center=mp.Vector3(0.35*self.sxy), size=mp.Vector3(y=0.8*self.sxy))
        thin_x_flux = sim.add_dft_fields([mp.Ez], self.fcen, self.fcen, 1, where=thin_x_volume)
        thin_y_volume = mp.Volume(center=mp.Vector3(y=0.25*self.sxy), size=mp.Vector3(x=self.sxy))
        thin_y_flux = sim.add_flux(self.fcen, self.fcen, 1, mp.FluxRegion(volume=thin_y_volume))

        sim.run(until_after_sources=100)

        # test proper collapsing of degenerate dimensions in HDF5 files and arrays
        thin_x_array = sim.get_dft_array(thin_x_flux, mp.Ez, 0)
        thin_y_array = sim.get_dft_array(thin_y_flux, mp.Ez, 0)
        np.testing.assert_equal(thin_x_array.ndim, 1)
        np.testing.assert_equal(thin_y_array.ndim, 1)

        sim.output_dft(thin_x_flux, os.path.join(temp_dir, 'thin-x-flux'))
        sim.output_dft(thin_y_flux, os.path.join(temp_dir, 'thin-y-flux'))

        with h5py.File(os.path.join(temp_dir, 'thin-x-flux.h5'), 'r') as thin_x:
            thin_x_h5 = mp.complexarray(thin_x['ez_0.r'][()], thin_x['ez_0.i'][()])

        with h5py.File(os.path.join(temp_dir, 'thin-y-flux.h5'), 'r') as thin_y:
            thin_y_h5 = mp.complexarray(thin_y['ez_0.r'][()], thin_y['ez_0.i'][()])

        np.testing.assert_allclose(thin_x_array, thin_x_h5)
        np.testing.assert_allclose(thin_y_array, thin_y_h5)

        # compare array data to HDF5 file content for fields and flux
        fields_arr = sim.get_dft_array(dft_fields, mp.Ez, 0)
        flux_arr = sim.get_dft_array(dft_flux, mp.Ez, 0)

        sim.output_dft(dft_fields, os.path.join(temp_dir, 'dft-fields'))
        sim.output_dft(dft_flux, os.path.join(temp_dir, 'dft-flux'))

        with h5py.File(os.path.join(temp_dir, 'dft-fields.h5'), 'r') as fields, h5py.File(os.path.join(temp_dir, 'dft-flux.h5'), 'r') as flux:
            exp_fields = mp.complexarray(fields['ez_0.r'][()], fields['ez_0.i'][()])
            exp_flux = mp.complexarray(flux['ez_0.r'][()], flux['ez_0.i'][()])

        np.testing.assert_allclose(exp_fields, fields_arr)
        np.testing.assert_allclose(exp_flux, flux_arr)

if __name__ == '__main__':
    if mp.am_master():
        temp_dir = tempfile.mkdtemp()
    else:
        temp_dir = None
    temp_dir = mp.comm.bcast(temp_dir, root=0)
    unittest.main()
    if mp.am_master():
        os.removedirs(temp_dir)
