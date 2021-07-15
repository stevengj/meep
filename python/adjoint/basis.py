import meep as mp
import numpy as np
from scipy import sparse
from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object, ), {'__slots__':
                                  ()})  # compatible with Python 2 and 3


#----------------------------------------------------------------------
# Basis is the abstract base class from which classes describing specific
# basis sets should inherit.
#----------------------------------------------------------------------
class Basis(ABC):
    """
    """
    def __init__(self,
                 rho_vector=None,
                 volume=None,
                 size=None,
                 center=mp.Vector3()):
        self.volume = volume if volume else mp.Volume(center=center, size=size)
        self.rho_vector = rho_vector

    def func(self):
        def _f(p):
            return self(p)

        return _f

    @abstractmethod
    def get_basis_vjp(self):
        raise NotImplementedError(
            "derived class must implement get_basis_vjp method")

    @abstractmethod
    def __call__(self, p=[0.0, 0.0]):
        raise NotImplementedError(
            "derived class must implement __call__() method")

    def set_rho_vector(self, rho_vector):
        self.rho_vector = rho_vector


# -------------------------------- #
# Bilinear Interpolation Basis class
# -------------------------------- #


class BilinearInterpolationBasis(Basis):
    ''' 
    Simple bilinear interpolation basis set.
    '''
    def __init__(self, resolution, symmetry=None, **kwargs):
        ''' 
        
        '''
        self.dim = 2

        super(BilinearInterpolationBasis, self).__init__(**kwargs)

        # Generate interpolation grid
        if symmetry is None or len(symmetry) == 0:
            self.symmetry = []
        else:
            self.symmetry = symmetry

        if mp.X in set(self.symmetry):
            self.Nx = int(resolution * self.volume.size.x / 2)
            self.rho_x = np.linspace(
                self.volume.center.x,
                self.volume.center.x + self.volume.size.x / 2, self.Nx)
            self.mirror_X = True
        else:
            self.Nx = int(resolution * self.volume.size.x)
            self.rho_x = np.linspace(
                self.volume.center.x - self.volume.size.x / 2,
                self.volume.center.x + self.volume.size.x / 2, self.Nx)
            self.mirror_X = False

        if mp.Y in set(self.symmetry):
            self.Ny = int(resolution * self.volume.size.y / 2)
            self.rho_y = np.linspace(
                self.volume.center.y,
                self.volume.center.y + self.volume.size.y / 2, self.Ny)
            self.mirror_Y = True
        else:
            self.Ny = int(resolution * self.volume.size.y)
            self.rho_y = np.linspace(
                self.volume.center.y - self.volume.size.y / 2,
                self.volume.center.y + self.volume.size.y / 2, self.Ny)
            self.mirror_Y = False

        self.num_design_params = self.Nx * self.Ny

        if self.rho_vector is None:
            self.rho_vector = np.ones((self.num_design_params, ))

    def __call__(self, p):
        x = 2 * self.volume.center.x - p.x if self.mirror_X and p.x < self.volume.center.x else p.x
        y = 2 * self.volume.center.y - p.y if self.mirror_Y and p.y < self.volume.center.y else p.y

        weights, interp_idx = self.get_bilinear_row(
            x, y, self.rho_x, self.rho_y)  # ignore z coordinate
        return np.dot(self.rho_vector[interp_idx], weights)

    def get_basis_vjp(self, dJ_deps, design_grid):
        ''' get vector jacobian product of interpolator'''

        dg_Nx, dg_Ny, Nz, Nf = dJ_deps.shape  # get important design grid dimensions
        x_grid = design_grid.x
        y_grid = design_grid.y
        z_grid = design_grid.z

        # take care of symmetries
        if self.mirror_X:
            dJ_deps = dJ_deps[int(dg_Nx / 2):, :, :, :] * 2
            x_grid = x_grid[int(dg_Nx / 2):]
        if self.mirror_Y:
            dJ_deps = dJ_deps[:, int(dg_Ny / 2):, :, :] * 2
            y_grid = y_grid[int(dg_Ny / 2):]

        dg_Nx, dg_Ny, Nz, Nf = dJ_deps.shape  # recalculate
        Nx, Ny = self.rho_x.size, self.rho_y.size  # get important interpolator dimensions

        # same interpolation matrix for all frequencies and all coordinates in Z direction
        A = self.gen_interpolation_matrix(self.rho_x, self.rho_y, x_grid,
                                          y_grid, z_grid)
        # TODO ditch the for loops
        dJ_dp = np.zeros((Nx * Ny, Nf))
        for fi in range(Nf):
            for zi in range(Nz):
                dJ_dp[:, fi] += np.matmul(
                    dJ_deps[:, :, zi, fi].reshape(dg_Nx * dg_Ny, order='C'), A)
        return dJ_dp

    def get_bilinear_coefficients(self, x, x1, x2, y, y1, y2):
        '''
        Calculates the bilinear interpolation coefficients for a single point at (x,y).
        Assumes that the user already knows the four closest points and provides the corresponding
        (x1,x2) and(y1,y2) coordinates.
        '''
        b11 = ((x - x2) * (y - y2)) / ((x1 - x2) * (y1 - y2))
        b12 = -((x - x2) * (y - y1)) / ((x1 - x2) * (y1 - y2))
        b21 = -((x - x1) * (y - y2)) / ((x1 - x2) * (y1 - y2))
        b22 = ((x - x1) * (y - y1)) / ((x1 - x2) * (y1 - y2))
        return [b11, b12, b21, b22]

    def get_bilinear_row(self, rx, ry, rho_x, rho_y):
        '''
        Calculates a vector of bilinear interpolation weights that can be used
        in an inner product with the neighboring function values, or placed
        inside of an interpolation matrix.
        '''

        Nx = rho_x.size
        Ny = rho_y.size

        # binary search in x direction to get x1 and x2
        xi2 = np.searchsorted(rho_x, rx, side='left')
        if xi2 <= 0:  # extrapolation (be careful!)
            xi1 = 0
            xi2 = 1
        elif xi2 >= Nx - 1:  # extrapolation (be careful!)
            xi1 = Nx - 2
            xi2 = Nx - 1
        else:
            xi1 = xi2 - 1

        x1 = rho_x[xi1]
        x2 = rho_x[xi2]

        # binary search in y direction to get y1 and y2
        yi2 = np.searchsorted(rho_y, ry, side='left')
        if yi2 <= 0:  # extrapolation (be careful!)
            yi1 = 0
            yi2 = 1
        elif yi2 >= Ny - 1:  # extrapolation (be careful!)
            yi1 = Ny - 2
            yi2 = Ny - 1
        else:
            yi1 = yi2 - 1

        y1 = rho_y[yi1]
        y2 = rho_y[yi2]

        # get weights
        weights = self.get_bilinear_coefficients(rx, x1, x2, ry, y1, y2)

        # get location of nearest neigbor interpolation points
        interp_idx = np.array([
            xi1 * Ny + yi1, xi1 * Ny + yi2, (xi2) * Ny + yi1, (xi2) * Ny + yi2
        ],
                              dtype=np.int64)

        return weights, interp_idx

    def gen_interpolation_matrix(self, rho_x, rho_y, rho_x_interp,
                                 rho_y_interp, rho_z_interp):
        '''
        Generates a bilinear interpolation matrix.
        
        Arguments:
        rho_x ................ [N,] numpy array - original x array mapping to povided data
        rho_y ................ [N,] numpy array - original y array mapping to povided data
        rho_x_interp ......... [N,] numpy array - new x array mapping to desired interpolated data
        rho_y_interp ......... [N,] numpy array - new y array mapping to desired interpolated data

        Returns:
        A .................... [N,M] sparse matrix - interpolation matrix
        '''

        Nx = rho_x.size
        Ny = rho_y.size
        Nx_interp = np.array(rho_x_interp).size
        Ny_interp = np.array(rho_y_interp).size
        Nz_interp = np.array(rho_y_interp).size

        input_dimension = Nx * Ny
        output_dimension = Nx_interp * Ny_interp

        interp_weights = np.zeros(4 * output_dimension)
        row_ind = np.zeros(4 * output_dimension, dtype=np.int64)
        col_ind = np.zeros(4 * output_dimension, dtype=np.int64)

        ri = 0
        for rx in rho_x_interp:
            for ry in rho_y_interp:
                # get weights
                weights, interp_idx = self.get_bilinear_row(
                    rx, ry, rho_x, rho_y)

                # populate sparse matrix vectors
                interp_weights[4 * ri:4 * (ri + 1)] = weights
                row_ind[4 * ri:4 * (ri + 1)] = np.array([ri, ri, ri, ri],
                                                        dtype=np.int64)
                col_ind[4 * ri:4 * (ri + 1)] = interp_idx

                ri += 1

        # From matrix vectors, populate the sparse matrix
        A = sparse.coo_matrix((interp_weights, (row_ind, col_ind)),
                              shape=(output_dimension, input_dimension))

        return A
