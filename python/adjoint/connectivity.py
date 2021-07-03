"""
Connectivity constraint inspired by https://link.springer.com/article/10.1007/s00158-016-1459-5
Solve and find adjoint gradients for [-div (k grad) + alpha^2 (1-rho)k + alpha0^2]T=0.
BC: Dirichlet T0 on last slice rho[-Nx*Ny:], 0 outside the first slice, and Neumann on sides.
Mo Chen <mochen@mit.edu>
"""

import numpy as np
from scipy.sparse.linalg import *
from scipy.sparse import kron, diags, csr_matrix, eye, csc_matrix, lil_matrix
from scipy.linalg import norm
import matplotlib.pyplot as plt
class ConnectivityConstraint(object):
    def __init__(self, nx, ny, nz, k0=1000, T0=1000, zeta=0, sp_solver=spsolve, alpha=None, alpha0=None, thresh=0.1, p=1):
        #zeta is to prevent singularity when damping is zero; with damping, zeta should be zero
        self.nx, self.ny, self.nz= nx, ny, nz
        self.n = nx*ny*nz
        self.m = nx*ny*(nz-1)
        self.solver = sp_solver
        self.k0, self.T0, self.zeta = k0, T0, zeta
        self.thresh = thresh
        self.p = p

        #default alpha and alpha0
        if alpha != None:
            self.alpha=alpha
        else:
            self.alpha = 0.1*min(1/nx, 1/ny, 1/nz)
        if alpha0 != None:
            self.alpha0 = alpha0
        else:
            self.alpha0 = -np.log(thresh)/min(nx, nz)


    def forward(self, rho_vector):
        self.rho_vector = rho_vector
        # gradient and -div operator
        gx = diags([-1,1], [0,1], shape=(self.nx-1, self.nx), format='csr')
        dx = gx.copy().transpose()
        gy = diags([-1,1], [0,1], shape=(self.ny-1, self.ny), format='csr')
        dy = gy.copy().transpose()
        gz = diags([1,-1], [0, -1], shape=(self.nz, self.nz), format='csr')
        dz = diags([1,-1], [0, 1], shape=(self.nz-1, self.nz), format='csr')

        # kron product for 2D
        Ix, Iy, Iz = eye(self.nx), eye(self.ny), eye(self.nz-1)
        self.gx, self.gy, self.gz = kron(Iz, kron(Iy, gx)), kron(Iz, kron(gy, Ix)), kron(gz, kron(Iy,Ix))
        self.dx, self.dy, self.dz = kron(Iz, kron(Iy, dx)), kron(Iz, kron(dy, Ix)), kron(dz, kron(Iy, Ix))

        #conductivity based on rho
        rho_pad = np.reshape(rho_vector, (self.nz, self.ny, self.nx))
        rhox = np.array([0.5*(rho_pad[k, j, i]+rho_pad[k, j, i+1]) for k in range(self.nz-1) for j in range(self.ny) for i in range(self.nx-1)])
        self.rhox = rhox
        rhoy = np.array([0.5*(rho_pad[k, j, i]+rho_pad[k, j+1, i]) for k in range(self.nz-1) for j in range(self.ny-1) for i in range(self.nx)])
        self.rhoy = rhoy
        rhoz = np.array([0.5*(rho_pad[k, j, i]+rho_pad[k+1, j, i]) for k in range(self.nz-1) for j in range(self.ny) for i in range(self.nx)])
        rhoz = np.insert(rhoz, [0]*self.nx*self.ny, 0)#0 outside first row
        self.rhoz = rhoz
        kx, ky, kz = diags((self.zeta+(1-self.zeta)*rhox)*self.k0), diags((self.zeta+(1-self.zeta)*rhoy)*self.k0), diags((self.zeta+(1-self.zeta)*rhoz)*self.k0)
        self.kx, self.ky, self.kz = kx, ky, kz
        #matrices in x, y, z
        self.Lx, self.Ly, self.Lz = self.dx * kx * self.gx, self.dy * ky * self.gy, self.dz * kz * self.gz

        # Dirichlet condition on the last row becomes term on the RHS
        Bz = csc_matrix(self.Lz)[:,-self.nx*self.ny:]
        rhs = -self.T0*Bz.sum(axis=1)
        self.rhs=rhs

        #LHS operator after moving the boundary term to the RHS
        eq = self.Lz[:, :-self.nx*self.ny]+self.Lx+self.Ly
        self.eq=eq
        #add damping
        damping = self.k0*self.alpha**2*diags(1-rho_vector[:-self.nx*self.ny]) + diags([self.alpha0**2], shape=(self.m, self.m))
        self.A = eq + damping
        self.damping = damping
        self.T = self.solver(csr_matrix(self.A), rhs)
        #exclude last row of rho and calculate weighted average of temperature
        self.rho_vec = rho_vector[:-self.nx*self.ny]

        self.Td_p = (self.T0 - self.T)**self.p
        self.Td = (sum(self.Td_p * self.rho_vec)/sum(self.rho_vec))**(1/self.p)

        return self.Td

    def adjoint(self):
        T_p1 = -(self.T-self.T0) ** (self.p-1)
        dg_dT = self.Td**(1-self.p) * (T_p1*self.rho_vec)/sum(self.rho_vec)
        return self.solver(csr_matrix(self.A.transpose()), dg_dT)
    def calculate_grad(self):
        dg_dp = np.zeros(self.n)
        dg_dp[:-self.nx*self.ny] = (self.Td_p*sum(self.rho_vec))/sum(self.rho_vec)**2
        dg_dp[:-self.nx*self.ny] = dg_dp[:-self.nx*self.ny] - sum(self.Td_p*self.rho_vec)/sum(self.rho_vec)**2
        dg_dp = self.Td ** (1-self.p) * dg_dp / self.p

        dAx = lil_matrix((self.m, self.n))
        gxT = np.reshape(self.gx * self.T, (-1,1))
        drhox = kron(eye(self.nz-1), kron(eye(self.ny), diags([0.5,0.5], [0, 1], shape=[self.nx-1,self.nx])))
        dAx[:, :-self.nx*self.ny] =  (1-self.zeta)*self.k0*lil_matrix(self.dx * drhox.multiply(gxT)) #element-wise product

        dAy = lil_matrix((self.m, self.n))
        gyT = np.reshape(self.gy * self.T, (-1,1))
        drhoy = kron(kron(eye(self.nz-1), diags([0.5,0.5], [0, 1], shape=[self.ny-1,self.ny])), eye(self.nx))
        dAy[:, :-self.nx*self.ny] =  (1-self.zeta)*self.k0*lil_matrix(self.dy * drhoy.multiply(gyT)) #element-wise product

        Tz = np.pad(self.T, (0, self.nx*self.ny), 'constant', constant_values=self.T0)
        gzTz = np.reshape(self.gz * Tz, (-1,1))
        drhoz = diags([0.5,0.5], [0, -1], shape=[self.nz,self.nz], format="lil")
        drhoz[0,0]=0
        drhoz = kron(drhoz,eye(self.nx*self.ny))
        dAz =  (1-self.zeta)*self.k0*self.dz * drhoz.multiply(gzTz)

        d_damping = self.k0*self.alpha**2*diags(-self.T, shape=(self.m, self.n))

        self.grad = dg_dp + self.adjoint().reshape(1, -1) * csr_matrix( - dAz - dAx - dAy - d_damping)
        return self.grad[0]

    def __call__(self, rho_vector):
        Td = self.forward(rho_vector)
        grad = self.calculate_grad()
        return Td-self.thresh*self.T0, grad
    def calculate_fd_grad(self, rho_vector, num, db=1e-4):
        fdidx = np.random.choice(self.n, num)
        fdgrad = []
        for k in fdidx:
            rho_vector[k]+=db
            fp = self.forward(rho_vector)
            rho_vector[k]-=2*db
            fm = self.forward(rho_vector)
            fdgrad.append((fp-fm)/(2*db))
            rho_vector[k]+=db
        return fdidx, fdgrad
    def calculate_all_fd_grad(self, rho_vector, db=1e-4):
        fdgrad = []
        for k in range(self.n):
            rho_vector[k]+=db
            fp = self.forward(rho_vector)
            rho_vector[k]-=2*db
            fm = self.forward(rho_vector)
            fdgrad.append((fp-fm)/(2*db))
            rho_vector[k]+=db
        return range(self.n), np.array(fdgrad)
