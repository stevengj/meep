"""
General filter functions to be used in other projection and morphological transform routines.
"""

import numpy as np
from autograd import numpy as npa
import meep as mp


'''
# ------------------------------------------------------------------------------------ #
Helper functions to be used by other routines
'''
def _centered(arr, newshape):
    '''Helper function that reformats the padded array of the fft filter operation.

    Borrowed from scipy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270
    '''
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _edge_pad(arr, pad):
    
    # fill sides
    left = npa.tile(arr[0,:],(pad[0][0],1)) # left side
    right = npa.tile(arr[-1,:],(pad[0][1],1)) # right side
    top = npa.tile(arr[:,0],(pad[1][0],1)).transpose() # top side
    bottom = npa.tile(arr[:,-1],(pad[1][1],1)).transpose() # bottom side)
    
    # fill corners
    top_left = npa.tile(arr[0,0], (pad[0][0],pad[1][0])) # top left
    top_right = npa.tile(arr[-1,0], (pad[0][1],pad[1][0])) # top right
    bottom_left = npa.tile(arr[0,-1], (pad[0][0],pad[1][1])) # bottom left
    bottom_right = npa.tile(arr[-1,-1], (pad[0][1],pad[1][1])) # bottom right
    
    out = npa.concatenate((
        npa.concatenate((top_left,top,top_right)),
        npa.concatenate((left,arr,right)),
        npa.concatenate((bottom_left,bottom,bottom_right))    
    ),axis=1)
    
    return out

def _zero_pad(arr, pad):
    
    # fill sides
    left = npa.tile(0,(pad[0][0],arr.shape[1])) # left side
    right = npa.tile(0,(pad[0][1],arr.shape[1])) # right side
    top = npa.tile(0,(arr.shape[0],pad[1][0])) # top side
    bottom = npa.tile(0,(arr.shape[0],pad[1][1])) # bottom side
    
    # fill corners
    top_left = npa.tile(0, (pad[0][0],pad[1][0])) # top left
    top_right = npa.tile(0, (pad[0][1],pad[1][0])) # top right
    bottom_left = npa.tile(0, (pad[0][0],pad[1][1])) # bottom left
    bottom_right = npa.tile(0, (pad[0][1],pad[1][1])) # bottom right
    
    out = npa.concatenate((
        npa.concatenate((top_left,top,top_right)),
        npa.concatenate((left,arr,right)),
        npa.concatenate((bottom_left,bottom,bottom_right))    
    ),axis=1)
    
    return out

'''
# ------------------------------------------------------------------------------------ #
Parameter filters
'''

def simple_2d_filter(x,kernel,Lx,Ly,resolution,symmetries=[]):
    """A simple 2d filter algorithm that is differentiable with autograd.
    Uses a 2D fft approach since it is typically faster and preserves the shape
    of the input and output arrays.
    
    The ffts pad the operation to prevent any circular convolution garbage.

    Parameters
    ----------
    x : array_like (2D)
        Input array to be filtered. Must be 2D.
    kernel : array_like (2D)
        Filter kernel (before the DFT). Must be same size as `x`
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    symmetries : list
        Symmetries to impose on the parameter field (either mp.X, mp.Y, or both)
    
    Returns
    -------
    array_like (2D)
        The output of the 2d convolution.
    """
    # Get 2d parameter space shape
    Nx = int(Lx*resolution)
    Ny = int(Ly*resolution)
    (kx,ky) = kernel.shape
    
    # Adjust parameter space shape for symmetries
    if mp.X in symmetries:
        Nx = int(Nx/2)
    if mp.Y in symmetries:
        Ny = int(Ny/2)
    
    # Ensure the input is 2D
    x = x.reshape(Nx,Ny)
    
    # Perform the required reflections for symmetries
    if mp.X in symmetries:
        if kx % 2 == 1:
            x = npa.concatenate((x,x[-1,:][None,:],x[::-1,:]), axis=0)
        else:
            x = npa.concatenate((x,x[::-1,:]), axis=0)
    if mp.Y in symmetries:
        if ky % 2 == 1:
            x = npa.concatenate((x[:,::-1],x[:,-1][:,None],x), axis=1)
        else:
            x = npa.concatenate((x[:,::-1],x), axis=1)
    
    # pad the kernel and input to avoid circular convolution and
    # to ensure boundary conditions are met.
    kernel = _zero_pad(kernel,((kx,kx),(ky,ky)))
    x = _edge_pad(x,((kx,kx),(ky,ky)))
    
    # Transform to frequency domain for fast convolution
    H = npa.fft.fft2(kernel)
    X = npa.fft.fft2(x)
    
    # Convolution (multiplication in frequency domain)
    Y = H * X
    
    # We need to fftshift since we padded both sides if each dimension of our input and kernel.
    y = npa.fft.fftshift(npa.real(npa.fft.ifft2(Y)))
    
    # Remove all the extra padding
    y = _centered(y,(kx,ky))
    
    # Remove the added symmetry domains
    if mp.X in symmetries:
        y = y[0:Nx,:]
    if mp.Y in symmetries:
        y = y[:,-Ny:]
    
    return y 

def cylindrical_filter(x,radius,Lx,Ly,resolution,symmetries=[]):
    '''A uniform cylindrical filter [1]. Typically allows for sharper transitions. 
    
    Parameters
    ----------
    x : array_like (2D)
        Design parameters
    radius : float
        Filter radius (in "meep units")
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    symmetries : list
        Symmetries to impose on the parameter field (either mp.X, mp.Y, or both)

    Returns
    -------
    array_like (2D)
        Filtered design parameters.
    
    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in 
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''    
    # Get 2d parameter space shape
    Nx = int(Lx*resolution)
    Ny = int(Ly*resolution)
    
    # Formulate grid over entire design region
    xv, yv = np.meshgrid(np.linspace(-Lx/2,Lx/2,Nx), np.linspace(-Ly/2,Ly/2,Ny), sparse=True, indexing='ij')
    
    # Calculate kernel
    kernel = np.where(np.abs(xv ** 2 + yv ** 2) <= radius**2,1,0).T
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel.flatten()) # Normalize the filter
    
    # Filter the response
    y = simple_2d_filter(x,kernel,Lx,Ly,resolution,symmetries)
    
    return y

def conic_filter(x,radius,Lx,Ly,resolution,symmetries=[]):
    '''A linear conic filter, also known as a "Hat" filter in the literature [1].
    
    Parameters
    ----------
    x : array_like (2D)
        Design parameters
    radius : float
        Filter radius (in "meep units")
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    symmetries : list
        Symmetries to impose on the parameter field (either mp.X, mp.Y, or both)

    Returns
    -------
    array_like (2D)
        Filtered design parameters.
    
    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in 
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''
    # Get 2d parameter space shape
    Nx = int(Lx*resolution)
    Ny = int(Ly*resolution)
    
    # Formulate grid over entire design region
    xv, yv = np.meshgrid(np.linspace(-Lx/2,Lx/2,Nx), np.linspace(-Ly/2,Ly/2,Ny), sparse=True, indexing='ij')
    
    # Calculate kernel
    kernel = np.where(np.abs(xv ** 2 + yv ** 2) <= radius**2,(1-np.sqrt(abs(xv ** 2 + yv ** 2))/radius),0)
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel.flatten()) # Normalize the filter
    
    # Filter the response
    y = simple_2d_filter(x,kernel,Lx,Ly,resolution,symmetries)
    
    return y


def gaussian_filter(x,sigma,Lx,Ly,resolution,symmetries=[]):
    '''A simple gaussian filter of the form exp(-x **2 / sigma ** 2) [1].
    
    Parameters
    ----------
    x : array_like (2D)
        Design parameters
    sigma : float
        Filter radius (in "meep units")
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)
    symmetries : list
        Symmetries to impose on the parameter field (either mp.X, mp.Y, or both)

    Returns
    -------
    array_like (2D)
        Filtered design parameters.
    
    References
    ----------
    [1] Wang, E. W., Sell, D., Phan, T., & Fan, J. A. (2019). Robust design of 
    topology-optimized metasurfaces. Optical Materials Express, 9(2), 469-482.
    '''
    # Get 2d parameter space shape
    Nx = int(Lx*resolution)
    Ny = int(Ly*resolution)

    gaussian = lambda x,sigma: np.exp(-x**2/sigma**2)
    
    # Formulate grid over entire design region
    xv = np.linspace(-Lx/2,Lx/2,Nx)
    yv = np.linspace(-Ly/2,Ly/2,Ny)
    
    # Calculate kernel
    kernel = np.outer(gaussian(xv, sigma), gaussian(yv, sigma)) # Gaussian filter kernel
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel.flatten()) # Normalize the filter
    
    # Filter the response
    y = simple_2d_filter(x,kernel,Lx,Ly,resolution,symmetries)
    
    return y

'''
# ------------------------------------------------------------------------------------ #
Erosion and dilation (morphological) operators
'''
    
def exponential_erosion(x,radius,beta,Lx,Ly,resolution):
    ''' Performs and exponential erosion operation.
    
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)

    Returns
    -------
    array_like
        Eroded design parameters.
    
    References
    ----------
    [1] Sigmund, O. (2007). Morphology-based black and white filters for topology optimization. 
    Structural and Multidisciplinary Optimization, 33(4-5), 401-424.
    [2] Schevenels, M., & Sigmund, O. (2016). On the implementation and effectiveness of 
    morphological close-open and open-close filters for topology optimization. Structural 
    and Multidisciplinary Optimization, 54(1), 15-21.
    '''
    
    x_hat = npa.exp(beta*(1-x))
    return 1 - npa.log(cylindrical_filter(x_hat,radius,Lx,Ly,resolution).flatten()) / beta

def exponential_dilation(x,radius,beta,Lx,Ly,resolution):
    ''' Performs a exponential dilation operation.
    
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)

    Returns
    -------
    array_like
        Dilated design parameters.
    
    References
    ----------
    [1] Sigmund, O. (2007). Morphology-based black and white filters for topology optimization. 
    Structural and Multidisciplinary Optimization, 33(4-5), 401-424.
    [2] Schevenels, M., & Sigmund, O. (2016). On the implementation and effectiveness of 
    morphological close-open and open-close filters for topology optimization. Structural 
    and Multidisciplinary Optimization, 54(1), 15-21.
    '''
    
    x_hat = npa.exp(beta*x)
    return npa.log(cylindrical_filter(x_hat,radius,Lx,Ly,resolution).flatten()) / beta

def heaviside_erosion(x,radius,beta,Lx,Ly,resolution):
    ''' Performs a heaviside erosion operation.
    
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)

    Returns
    -------
    array_like
        Eroded design parameters.
    
    References
    ----------
    [1] Guest, J. K., Prévost, J. H., & Belytschko, T. (2004). Achieving minimum length scale in topology 
    optimization using nodal design variables and projection functions. International journal for 
    numerical methods in engineering, 61(2), 238-254.
    '''
    
    x_hat = cylindrical_filter(x,radius,Lx,Ly,resolution).flatten()
    return npa.exp(-beta*(1-x_hat)) + npa.exp(-beta)*(1-x_hat)

def heaviside_dilation(x,radius,beta,Lx,Ly,resolution):
    ''' Performs a heaviside dilation operation.
    
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)

    Returns
    -------
    array_like
        Dilated design parameters.
    
    References
    ----------
    [1] Guest, J. K., Prévost, J. H., & Belytschko, T. (2004). Achieving minimum length scale in topology 
    optimization using nodal design variables and projection functions. International journal for 
    numerical methods in engineering, 61(2), 238-254.
    '''
    
    x_hat = cylindrical_filter(x,radius,Lx,Ly,resolution).flatten()
    return 1 - npa.exp(-beta*x_hat) + npa.exp(-beta)*x_hat

def geometric_erosion(x,radius,alpha,Lx,Ly,resolution):
    ''' Performs a geometric erosion operation.
    
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)

    Returns
    -------
    array_like
        Eroded design parameters.
    
    References
    ----------
    [1] Svanberg, K., & Svärd, H. (2013). Density filters for topology optimization based on the 
    Pythagorean means. Structural and Multidisciplinary Optimization, 48(5), 859-875.
    '''
    x_hat = npa.log(x + alpha)
    return npa.exp(cylindrical_filter(x_hat,radius,Lx,Ly,resolution)).flatten() - alpha

def geometric_dilation(x,radius,alpha,Lx,Ly,resolution):
    ''' Performs a geometric dilation operation.
    
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)

    Returns
    -------
    array_like
        Dilated design parameters.
    
    References
    ----------
    [1] Svanberg, K., & Svärd, H. (2013). Density filters for topology optimization based on the 
    Pythagorean means. Structural and Multidisciplinary Optimization, 48(5), 859-875.
    '''

    x_hat = npa.log(1 - x + alpha)
    return -npa.exp(cylindrical_filter(x_hat,radius,Lx,Ly,resolution)).flatten() + alpha + 1

def harmonic_erosion(x,radius,alpha,Lx,Ly,resolution):
    ''' Performs a harmonic erosion operation.
    
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)

    Returns
    -------
    array_like
        Eroded design parameters.
    
    References
    ----------
    [1] Svanberg, K., & Svärd, H. (2013). Density filters for topology optimization based on the 
    Pythagorean means. Structural and Multidisciplinary Optimization, 48(5), 859-875.
    '''
    
    x_hat = 1 / (x + alpha)
    return 1 / cylindrical_filter(x_hat,radius,Lx,Ly,resolution).flatten() - alpha

def harmonic_dilation(x,radius,alpha,Lx,Ly,resolution):
    ''' Performs a harmonic dilation operation.
    
    Parameters
    ----------
    x : array_like
        Design parameters
    radius : float
        Filter radius (in "meep units")
    beta : float
        Thresholding parameter
    Lx : float
        Length of design region in X direction (in "meep units")
    Ly : float
        Length of design region in Y direction (in "meep units")
    resolution : int
        Resolution of the design grid (not the meep simulation resolution)

    Returns
    -------
    array_like
        Dilated design parameters.
    
    References
    ----------
    [1] Svanberg, K., & Svärd, H. (2013). Density filters for topology optimization based on the 
    Pythagorean means. Structural and Multidisciplinary Optimization, 48(5), 859-875.
    '''
    
    x_hat = 1 / (1 - x + alpha)
    return 1 - 1 / cylindrical_filter(x_hat,radius,Lx,Ly,resolution).flatten() + alpha

'''
# ------------------------------------------------------------------------------------ #
Projection filters
'''

def tanh_projection(x,beta,eta):
    '''Projection filter that thresholds the input parameters between 0 and 1. Typically
    the "strongest" projection.

    Parameters
    ----------
    x : array_like
        Design parameters
    beta : float
        Thresholding parameter (0 to infinity). Dictates how "binary" the output will be.
    eta: float
        Threshold point (0 to 1)  

    Returns
    -------
    array_like
        Projected and flattened design parameters.
    References
    ----------
    [1] Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and robust 
    formulations in topology optimization. Structural and Multidisciplinary Optimization, 43(6), 767-784.
    '''
    
    return (npa.tanh(beta*eta) + npa.tanh(beta*(x-eta))) / (npa.tanh(beta*eta) + npa.tanh(beta*(1-eta)))

def heaviside_projection(x, beta):
    '''Projection filter that thresholds the input parameters between 0 and 1.

    Parameters
    ----------
    x : array_like
        Design parameters
    beta : float
        Thresholding parameter (0 to infinity). Dictates how "binary" the output will be.
    eta: float
        Threshold point (0 to 1)  

    Returns
    -------
    array_like
        Projected and flattened design parameters.
    
    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in 
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''
    
    case1 = eta*npa.exp(-beta*(eta-x)/eta) - (eta-x)*npa.exp(-beta)
    case2 = 1 - (1-eta)*npa.exp(-beta*(x-eta)/(1-eta)) - (eta-x)*npa.exp(-beta)
    return npa.where(x < eta,case1,case2)

'''
# ------------------------------------------------------------------------------------ #
Length scale operations
'''

def get_threshold_wang(delta,sigma):
    '''Calculates the threshold point according to the gaussian filter radius (`sigma`) and
    the perturbation parameter (`sigma`) needed to ensure the proper length
    scale and morphological transformation according to Wang et. al. [2].
    
    Parameters
    ----------
    sigma : float
        Smoothing radius (in meep units)
    delta : float
        Perturbation parameter (in meep units)
    
    Returns
    -------
    float
        Threshold point (`eta`)
    
    References
    ----------
    [1] Wang, F., Jensen, J. S., & Sigmund, O. (2011). Robust topology optimization of 
    photonic crystal waveguides with tailored dispersion properties. JOSA B, 28(3), 387-397.
    [2] Wang, E. W., Sell, D., Phan, T., & Fan, J. A. (2019). Robust design of 
    topology-optimized metasurfaces. Optical Materials Express, 9(2), 469-482.
    '''
    
    return 0.5 - special.erf(delta/sigma)

def get_eta_from_conic(b,R):
    ''' Extracts the eroded threshold point (`eta_e`) for a conic filter given the desired
    minimum length (`b`) and the filter radius (`R`). This only works for conic filters.
    
    Note that the units for `b` and `R` can be arbitrary so long as they are consistent.
    
    Results in paper were thresholded using a "tanh" Heaviside projection.
    
    Parameters
    ----------
    b : float
        Desired minimum length scale.
    R : float
        Conic filter radius
    
    Returns
    -------
    float
        The eroded threshold point (1-eta)
    
    References
    ----------
    [1] Qian, X., & Sigmund, O. (2013). Topological design of electromechanical actuators with 
    robustness toward over-and under-etching. Computer Methods in Applied 
    Mechanics and Engineering, 253, 237-251.
    [2] Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and 
    robust formulations in topology optimization. Structural and Multidisciplinary 
    Optimization, 43(6), 767-784.
    [3] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in 
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''
    
    norm_length = b/R
    if norm_length < 0:
        eta_e = 0
    elif norm_length < 1:
        eta_e = 0.25 * norm_length ** 2 + 0.5
    elif norm_length < 2:
        eta_e = -0.25 * norm_length ** 2 + norm_length
    else:
        eta_e = 1
    return eta_e

def get_conic_radius_from_eta_e(b,eta_e):
    """Calculates the corresponding filter radius given the minimum length scale (b)
    and the desired eroded threshold point (eta_e).
    
    Parameters
    ----------
    b : float
        Desired minimum length scale.
    eta_e : float
        Eroded threshold point (1-eta)
    
    Returns
    -------
    float
        Conic filter radius.
    
    References
    ----------
    [1] Qian, X., & Sigmund, O. (2013). Topological design of electromechanical actuators with 
    robustness toward over-and under-etching. Computer Methods in Applied 
    Mechanics and Engineering, 253, 237-251.
    [2] Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and 
    robust formulations in topology optimization. Structural and Multidisciplinary 
    Optimization, 43(6), 767-784.
    [3] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in 
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    """
    if (eta_e >= 0.5) and (eta_e < 0.75):
        return b / (2*np.sqrt(eta_e-0.5))
    elif (eta_e >= 0.75) and (eta_e <= 1):
        return b / (2-2*np.sqrt(1-eta_e))
    else:
        raise ValueError("The erosion threshold point (eta_e) must be between 0.5 and 1.")

def indicator_solid(x,c,filter_f,threshold_f,resolution):
    '''Calculates the indicator function for the void phase needed for minimum length optimization [1].
    
    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_e : float
        Erosion threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.
    
    Returns
    -------
    array_like
        Indicator value
    
    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by 
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''
    
    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)
    gradient_filtered_field = npa.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0]*resolution) ** 2 + (gradient_filtered_field[1]*resolution) ** 2
    if grad_mag.ndim != 2:
        raise ValueError("The gradient fields must be 2 dimensional. Check input array and filter functions.")
    I_s = design_field * npa.exp(-c * grad_mag)
    return I_s

def constraint_solid(x,c,eta_e,filter_f,threshold_f,resolution):
    '''Calculates the constraint function of the solid phase needed for minimum length optimization [1].
    
    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_e : float
        Erosion threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.
    
    Returns
    -------
    float
        Constraint value
    
    Example
    -------
    >> g_s = constraint_solid(x,c,eta_e,filter_f,threshold_f) # constraint
    >> g_s_grad = grad(constraint_solid,0)(x,c,eta_e,filter_f,threshold_f) # gradient
    
    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by 
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''
    
    filtered_field = filter_f(x)
    I_s = indicator_solid(x.reshape(filtered_field.shape),c,filter_f,threshold_f,resolution).flatten()
    return npa.mean(I_s * npa.minimum(filtered_field.flatten()-eta_e,0)**2)

def indicator_void(x,c,filter_f,threshold_f,resolution):
    '''Calculates the indicator function for the void phase needed for minimum length optimization [1].
    
    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_d : float
        Dilation threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.
    
    Returns
    -------
    array_like
        Indicator value
    
    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by 
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''
    
    filtered_field = filter_f(x).reshape(x.shape)
    design_field = threshold_f(filtered_field)
    gradient_filtered_field = npa.gradient(filtered_field)
    grad_mag = (gradient_filtered_field[0]*resolution) ** 2 + (gradient_filtered_field[1]*resolution) ** 2
    if grad_mag.ndim != 2:
        raise ValueError("The gradient fields must be 2 dimensional. Check input array and filter functions.")
    return (1 - design_field) * npa.exp(-c * grad_mag)

def constraint_void(x,c,eta_d,filter_f,threshold_f,resolution):
    '''Calculates the constraint function of the void phase needed for minimum length optimization [1].
    
    Parameters
    ----------
    x : array_like
        Design parameters
    c : float
        Decay rate parameter (1e0 - 1e8)
    eta_d : float
        Dilation threshold limit (0-1)
    filter_f : function_handle
        Filter function. Must be differntiable by autograd.
    threshold_f : function_handle
        Threshold function. Must be differntiable by autograd.
    
    Returns
    -------
    float
        Constraint value
    
    Example
    -------
    >> g_v = constraint_void(p,c,eta_d,filter_f,threshold_f) # constraint
    >> g_v_grad = tensor_jacobian_product(constraint_void,0)(p,c,eta_d,filter_f,threshold_f,g_s) # gradient
    
    References
    ----------
    [1] Zhou, M., Lazarov, B. S., Wang, F., & Sigmund, O. (2015). Minimum length scale in topology optimization by 
    geometric constraints. Computer Methods in Applied Mechanics and Engineering, 293, 266-282.
    '''
    
    filtered_field = filter_f(x)
    I_v = indicator_void(x.reshape(filtered_field.shape),c,filter_f,threshold_f,resolution).flatten()
    return npa.mean(I_v * npa.minimum(eta_d-filtered_field.flatten(),0)**2)

'''
# ------------------------------------------------------------------------------------ #
Material interpolators
'''

def SIMP(x,eps1,eps2,p=1):
    '''Interpolates the design parameters (0-1) into the desired permittivity contrast (eps1-eps2) [1,2].
    
    Parameters
    ----------
    x : array_like
        Filtered and thresholded design parameters (between 0 and 1)
    eps1 : scalar
        Permittivity of material 1 (must be less than eps2)
    eps2 : sca;ar
        Permittivity of material 2 (must be greater than eps1)
    p : scalar
        Penalization factor. Higher numbers encourage binarization. Default is 1; must be >= 1.
    
    Returns
    -------
    array_like
        Interpolated design variables
    
    References
    ----------

    [1] Andkjær, J., Johansen, V. E., Friis, K. S., & Sigmund, O. (2014). Inverse design of 
    nanostructured surfaces for color effects. JOSA B, 31(1), 164-174.
    [2] Christiansen, R. E., Vester-Petersen, J., Madsen, S. P., & Sigmund, O. (2019). A non-linear 
    material interpolation for design of metallic nano-particles using topology optimization. Computer 
    Methods in Applied Mechanics and Engineering, 343, 23-39.
    '''

    if eps1 >= eps2:
        raise ValueError("eps0 must be smaller than eps1.")
    return eps1 + x**p * (eps2-eps1)

def inverse_SIMP(x,eps1,eps2,p=1):
    '''Inversely interpolates the design parameters (0-1) into the desired permittivity contrast (eps1-eps2) [1,2].
    
    Parameters
    ----------
    x : array_like
        Filtered and thresholded design parameters (between 0 and 1)
    eps1 : scalar
        Permittivity of material 1 (must be less than eps2)
    eps2 : sca;ar
        Permittivity of material 2 (must be greater than eps1)
    p : scalar
        Penalization factor. Higher numbers encourage binarization. Default is 1; must be >= 1.
    
    Returns
    -------
    array_like
        Interpolated design variables
    
    References
    ----------
    [1] Wadbro, E., & Engström, C. (2015). Topology and shape optimization of plasmonic nano-antennas. 
    Computer Methods in Applied Mechanics and Engineering, 293, 155-169.
    [2] Christiansen, R. E., Vester-Petersen, J., Madsen, S. P., & Sigmund, O. (2019). A non-linear 
    material interpolation for design of metallic nano-particles using topology optimization. Computer 
    Methods in Applied Mechanics and Engineering, 343, 23-39.
    '''
    if eps1 >= eps2:
        raise ValueError("eps0 must be smaller than eps1.")
    return 1 / (1/eps1 + x**p*(1/eps2 - 1/eps1))

def metal_interpolation(x,n1,k1,n2,k2):
    '''Nonlinearly interpolates the design parameters (0-1) into the desired permittivity contrast (eps1-eps2).
    Especially useful for opimization of metal structures in dielectric/air environments. It avoids undesirable 
    zero crossings that may induce simulation instabilities.
    
    Parameters
    ----------
    x : array_like
        Filtered and thresholded design parameters (between 0 and 1)
    n1 : scalar
        Permittivity of material 1 (must be less than eps2)
    k1 : sca;ar
        Permittivity of material 2 (must be greater than eps1)
    
    Returns
    -------
    array_like
        Interpolated design variables
    
    References
    ----------
    [1] Christiansen, R. E., Vester-Petersen, J., Madsen, S. P., & Sigmund, O. (2019). A non-linear 
    material interpolation for design of metallic nano-particles using topology optimization. Computer 
    Methods in Applied Mechanics and Engineering, 343, 23-39.
    [2] Christiansen, R. E., Michon, J., Benzaouia, M., Sigmund, O., & Johnson, S. G. (2020). Inverse 
    design of nanoparticles for enhanced Raman scattering. Optics Express, 28(4), 4444-4462.
    '''

    n_bar = n1 + x*(n2-n1)
    k_bar = k1 + x*(k2-k1)
    return (n_bar**2 - k_bar**2) - 1j*(2*n_bar*k_bar)


def _eval_susceptibility(self,freq):
    sigma = np.expand_dims(Matrix(diag=self.sigma_diag,offdiag=self.sigma_offdiag),axis=0)
    if self.gamma == 0:
        return self.frequency*self.frequency / (self.frequency*self.frequency - freq*freq) * sigma
    else:
        return self.frequency*self.frequency / (self.frequency*self.frequency - freq*freq - 1j*self.gamma*freq) * sigma

def _eval_susceptibility(self,freq):
    sigma = np.expand_dims(Matrix(diag=self.sigma_diag,offdiag=self.sigma_offdiag),axis=0)
    if self.gamma == 0:
        return -self.frequency*self.frequency / (freq*(freq)) * sigma
    else:
        return -self.frequency*self.frequency / (freq*(freq + 1j*self.gamma)) * sigma

def _get_eps(eps0, susceptibilities, conductivity, freqs):
    # Clean the input
    if np.isscalar(freq):
        freqs = np.array(freq)[np.newaxis, np.newaxis, np.newaxis]
    else:
        freqs = np.squeeze(freq)
        freqs = freqs[:, np.newaxis, np.newaxis]

    # Initialize with instantaneous dielectric
    eps = eps0

    # Iterate through susceptibilities
    for i_sus, sus in enumerate(susceptibilities):
        epsmu = epsmu + _eval_susceptibility(sus,freqs)

    # Account for conductivity term (only multiply if nonzero to avoid unnecessary complex numbers)
    if np.count_nonzero(conductivity) > 0:
        epsmu = (1 + 1j/freqs * conductivity) * epsmu

    # Convert list matrix to 3D numpy array size [freqs,3,3]
    return np.squeeze(epsmu)

def _medium_interpolation(x,freqs,medium1,medium2):
    '''

    '''
    mat = medium_interpolation(x,freqs,medium1,medium2)

def medium_interpolation(x,freqs,medium1,medium2):
    '''
    TODO
    '''
    # get eps
    eps1 = medium1.epsilon
    eps2 = medium2.epsilon

    # get sus
    sus1 = medium1.E_susceptibilities.copy()
    sus2 = medium2.E_susceptibilities.copy()
    
    # interp the ep0
    epsilon = eps1 + x * (eps2-eps1)

    omega0 = []
    
    # interp the sus cond of medium1
    sigma1 = x
    for s in sus1:
        omega0.append(s.omega0)
        s.sigma_diag = sigma1*s.sigma_diag
    
    # interp the sus cond of medium2
    sigma2 = 1-x
    for s in sus2:
        omega0.append(s.omega0)
        s.sigma_diag = sigma2*s.sigma_diag
    
    # interp the damping term
    omega0 = np.mean(omega0)
    conductivity = epsilon * (1-epsilon) * omega0

    # build the final object
    return mp.Medium(epsilon=epsilon,E_susceptibilities=sus,D_conductivity=conductivity)

'''
# ------------------------------------------------------------------------------------ #
Multi-objective optimization tools.

A comprehensive survey of the various MOO methods is found in [1].

[1] Marler, R. T., & Arora, J. S. (2004). Survey of multi-objective optimization methods for 
engineering. Structural and multidisciplinary optimization, 26(6), 369-395.
'''

def scale_objective(F,Fmin,Fmax):
    return (F-Fmin)/(Fmax-Fmin)

def compromise_objective(F,Fmin,Fmax,w,p):
    '''
    References
    ----------
    [1] Sehlström, A. (2013). Multiobjective topology optimization: Tracing of Pareto-optimal 
    structures with respect to volume, compliance and fundamental eigenvalue (Master's thesis).
    '''

    F = npa.array(F)
    Fmin = npa.array(Fmin)
    Fmax = npa.array(Fmax)
    w = npa.array(w)
    return npa.sum( w * ((F-Fmin)/(Fmax-Fmin)),axis=0) ** (1/p)

def exponential_weighting():
    return

def weighted_product():
    return

def physical_programming(x):
    '''
    TODO
    References
    ----------
    [1] Messac, A. (1996). Physical programming-effective optimization for computational design. 
    AIAA journal, 34(1), 149-158.
    [2] Lin, J., Luo, Z., & Tong, L. (2010). A new multi-objective programming scheme for 
    topology optimization of compliant mechanisms. Structural and Multidisciplinary Optimization, 
    40(1-6), 241.
    '''
    return x

'''
# ------------------------------------------------------------------------------------ #
Misc. tools
'''

def gray_indicator(x):
    '''Calculates a measure of "grayness" according to [1].

    Lower numbers ( < 2%) indicate a good amount of binarization [1].
    
    Parameters
    ----------
    x : array_like
        Filtered and thresholded design parameters (between 0 and 1)
    
    Returns
    -------
    float
        Measure of "grayness" (in percent)
    
    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in 
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''
    return npa.mean(4 * x.flatten()) * (1-x.flatten()) * 100

