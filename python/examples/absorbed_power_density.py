import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import meep as mp
from meep.materials import SiO2

resolution = 50  # pixels/um

dpml = 1.0
pml_layers = [mp.PML(thickness=dpml)]

r = 1.0     # radius of cylinder
dair = 2.0  # air padding thickness

s = 2*(dpml+dair+r)
cell_size = mp.Vector3(s,s)

fcen = 1.0  # wavelength of 1.0 um

# is_integrated=True necessary for any planewave source extending into PML
sources = [mp.Source(mp.GaussianSource(fcen,fwidth=0.1*fcen,is_integrated=True),
                     center=mp.Vector3(-0.5*s+dpml),
                     size=mp.Vector3(0,s),
                     component=mp.Ez)]

symmetries = [mp.Mirror(mp.Y)]

geometry = [mp.Cylinder(material=SiO2,
                        center=mp.Vector3(),
                        radius=r,
                        height=mp.inf)]

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=pml_layers,
                    sources=sources,
                    k_point=mp.Vector3(),
                    symmetries=symmetries,
                    geometry=geometry)

dft_fields = sim.add_dft_fields([mp.Dz,mp.Ez], fcen, 0, 1, center=mp.Vector3(), size=mp.Vector3(2*r,2*r))

# closed box surrounding cylinder for computing total incoming flux
flux_box = sim.add_flux(fcen, 0, 1,
                        mp.FluxRegion(center=mp.Vector3(x=-r),size=mp.Vector3(0,2*r),weight=+1),
                        mp.FluxRegion(center=mp.Vector3(x=+r),size=mp.Vector3(0,2*r),weight=-1),
                        mp.FluxRegion(center=mp.Vector3(y=+r),size=mp.Vector3(2*r,0),weight=-1),
                        mp.FluxRegion(center=mp.Vector3(y=-r),size=mp.Vector3(2*r,0),weight=+1))

sim.run(until_after_sources=100)

plt.figure()
sim.plot2D()
plt.savefig('power_density_cell.png',dpi=150,bbox_inches='tight')

(x,y,z,w) = sim.get_array_metadata(dft_cell=dft_fields)
Dz = sim.get_dft_array(dft_fields,mp.Dz,0)
Ez = sim.get_dft_array(dft_fields,mp.Ez,0)
absorbed_power_density = 2*np.pi*fcen * np.imag(np.conj(Ez)*Dz)

plt.figure()
plt.pcolormesh(x,y,np.transpose(absorbed_power_density),cmap='inferno_r',shading='gouraud',vmin=0,vmax=np.amax(absorbed_power_density))
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal')
plt.title("absorbed power density")
plt.colorbar()
plt.savefig('power_density_map.png',dpi=150,bbox_inches='tight')

absorbed_power = np.sum(w*absorbed_power_density)
absorbed_flux = mp.get_fluxes(flux_box)[0]
err = abs(absorbed_power-absorbed_flux)/absorbed_flux
print("flux:, {} (dft_fields), {} (dft_flux), {} (error)".format(absorbed_power,absorbed_flux,err))
