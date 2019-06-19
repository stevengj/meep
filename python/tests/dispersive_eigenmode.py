
# dispersive_eigenmode.py - Tests the meep eigenmode features (eigenmode source,
# eigenmode decomposition, and get_eigenmode) with dispersive materials.
# TODO: 
#  * check materials with off diagonal components
#  * check magnetic profiles
#  * once imaginary component is supported, check that

from __future__ import division

import unittest
import meep as mp
import numpy as np
from meep import mpb
import h5py
import os


class TestDispersiveEigenmode(unittest.TestCase):

    # Directly calss the C++ chi1 routine
    def call_chi1(self,material,component,direction,omega):

        sim = mp.Simulation(cell_size=mp.Vector3(1,1,1),
                    default_material=material,
                    resolution=10)

        sim.init_sim()
        v3 = mp.py_v3_to_vec(sim.dimensions, mp.Vector3(0,0,0), sim.is_cylindrical)
        n = 1/np.sqrt(sim.structure.get_chi1inv(int(component),int(direction),v3,omega))
        return n

    # Pulls the "effective index" of a uniform, dispersive material
    # (i.e. the refractive index) using meep's get_eigenmode
    def simulate_meep(self,material,omega):
        
        sim = mp.Simulation(cell_size=mp.Vector3(2,2,2),
                            default_material=material,
                            resolution=20
                            )

        direction = mp.X
        where = mp.Volume(center=mp.Vector3(0,0,0),size=mp.Vector3(0,1,1))
        band_num = 1
        kpoint = mp.Vector3(2,0,0)
        sim.init_sim()
        em = sim.get_eigenmode(omega,direction,where,band_num,kpoint)
        neff_meep = np.squeeze(em.k.x) / np.squeeze(em.freq)

        return neff_meep
    
    # Pulls the "effective index" of a uniform, dispersive material
    # (i.e. the refractive index) using mpb
    def simulate_mpb(self,material,omega):
        ms = mpb.ModeSolver(
            geometry_lattice=mp.Lattice(size=mp.Vector3(0,2,2)),
            default_material=material,
            resolution=10,
            num_bands=1
        )
        k = ms.find_k(mp.NO_PARITY, omega, 1, 1, mp.Vector3(1), 1e-3, omega * 1,
                omega * 0.1, omega * 6)
        
        neff_mpb = k[0]/omega
        return neff_mpb
    
    # main test bed to check the new features
    def compare_meep_mpb(self,material,omega,component=0,direction=0):
        n = np.real(np.sqrt(material.epsilon(omega)[component,direction]))
        chi1 = self.call_chi1(material,mp.Ex,mp.X,omega)
        n_meep = self.simulate_meep(material,omega)
        # Let's wait to check this until we enable dispersive materials in MPB...
        #n_mpb = self.simulate_mpb(material,omega)

        # Check that the chi1 value matches the refractive index
        self.assertAlmostEqual(n,chi1, places=6)

        # Check that the chi1 value matches meep's get_eigenmode
        self.assertAlmostEqual(n,n_meep, places=6)
        
        # Check that the chi1 value matches mpb's get_eigenmode 
        #self.assertAlmostEqual(n,n_mpb, places=6)

    def test_chi1_routine(self):
        # This test checks the newly implemented get_chi1inv routines within the 
        # fields and structure classes by comparing their output to the 
        # python epsilon output.

        from meep.materials import Si, Ag, LiNbO3, Au
        
        # Check Silicon
        w0 = Si.valid_freq_range.min
        w1 = Si.valid_freq_range.max
        self.compare_meep_mpb(Si,w0)
        self.compare_meep_mpb(Si,w1)

        # Check Silver
        w0 = Ag.valid_freq_range.min
        w1 = Ag.valid_freq_range.max
        self.compare_meep_mpb(Ag,w0)
        self.compare_meep_mpb(Ag,w1)

        # Check Gold
        w0 = Au.valid_freq_range.min
        w1 = Au.valid_freq_range.max
        self.compare_meep_mpb(Au,w0)
        self.compare_meep_mpb(Au,w1)   

        # Check Lithium Niobate (X,X)
        w0 = LiNbO3.valid_freq_range.min
        w1 = LiNbO3.valid_freq_range.max
        self.compare_meep_mpb(LiNbO3,w0)
        self.compare_meep_mpb(LiNbO3,w1)
    
    def verify_output_and_slice(self,material,omega):
        # Since the slice routines average the diagonals, we need to do that too:
        n = np.mean(np.real(np.sqrt(material.epsilon(omega).diagonal())))
        
        sim = mp.Simulation(cell_size=mp.Vector3(2,2,2),
                            default_material=material,
                            resolution=20,
                            eps_averaging=False
                            )
        sim.init_sim()
        
        # Check to make sure the get_slice routine is working with omega
        n_slice = np.sqrt(np.min(sim.get_epsilon(omega)))
        self.assertAlmostEqual(n,n_slice, places=6)

        # Check to make sure h5 output is working with omega
        # NOTE: We'll add this test once h5 support is added
        #filename = 'dispersive_eigenmode-eps-000000.00.h5'
        #mp.output_epsilon(sim,omega=omega)
        #n_h5 = np.sqrt(np.min(h5py.File(filename, 'r')['eps']))
        #self.assertAlmostEqual(n,n_h5, places=6)
        #os.remove(filename)
    
    def test_get_with_dispersion(self):
        # This test checks the get_array_slice and output_fields method
        # with dispersive materials.

        from meep.materials import Si, Ag, LiNbO3, Au
        
        # Check Silicon
        w0 = Si.valid_freq_range.min
        w1 = Si.valid_freq_range.max
        self.verify_output_and_slice(Si,w0)
        self.verify_output_and_slice(Si,w1)

        # Check Silver
        w0 = Ag.valid_freq_range.min
        w1 = Ag.valid_freq_range.max
        self.verify_output_and_slice(Ag,w0)
        self.verify_output_and_slice(Ag,w1)

        # Check Gold
        w0 = Au.valid_freq_range.min
        w1 = Au.valid_freq_range.max
        self.verify_output_and_slice(Au,w0)
        self.verify_output_and_slice(Au,w1)       

        # Check Lithium Niobate
        w0 = LiNbO3.valid_freq_range.min
        w1 = LiNbO3.valid_freq_range.max
        #self.verify_output_and_slice(LiNbO3,w0)
        #self.verify_output_and_slice(LiNbO3,w1)
        

if __name__ == '__main__':
    unittest.main()
