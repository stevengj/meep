"""
General filter functions to be used in other projection and morphological transform routines.
"""

import numpy as np
import jax
from jax import numpy as npj
import meep as mp
from scipy import special
import jax.numpy.fft as fft


def centered(arr, newshape):
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


def atleast_3d(ary):
    if ary.ndim == 0:
        result = npj.expand_dims(ary, axis=(0, 1, 2))  #ary.reshape(1, 1, 1)
    elif ary.ndim == 1:
        result = npj.expand_dims(ary, axis=(1, 2))
    elif ary.ndim == 2:
        result = npj.expand_dims(ary, axis=(2))
    else:
        result = ary
    return result


def _centered(arr, newshape):
    '''Helper function that reformats the padded array of the fft filter operation.

    Borrowed from scipy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270
    '''
    # Return the center newshape portion of the array.
    newshape = npj.array(newshape)
    currshape = npj.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def meep_filter(x, kernel):
    '''
    General convolution for 1D, 2D, and 3D design spaces.

    Currently, Jax's XLA convolution kernel *always* uses 
    the overlap-add method, which is very costly for 
    the large arrays we typically convolve.

    In the meantime, we can do a manual n-dimensional
    FFT (also using jax). Note, however, that Jax
    doesn't support automatic padding (like numpy does)
    so we need to do the padding ourselves (in addition
    to the edge padding we use for boundary conditions).

    Note that we need to fftshift the output since we 
    padded *both* sides of our signals.

    Parameters
    ----------
    x : array_like (1D, 2D, or 3D)
        Input array to be filtered.
    kernel : array_like (1D, 2D, or 3D)
        Filter kernel (before the DFT). Must be same size as `x`
    
    Returns
    -------
    array_like (1D, 2D, or 3D)
        The output of the N-D convolution.
    '''
    # store shapes
    x_shape = x.shape
    k_shape = kernel.shape

    # edge pad
    npad = *((s, s) for s in x_shape),
    kernelp = npj.pad(kernel, npad, mode='edge')
    xp = npj.pad(x, npad, mode='edge')

    # convolve
    '''
    As is the case with any convolution, the filter will
    introduce a delay in every specified dimension. To 
    compensate for this delay, we can perform a zero-phase
    filter operation. We first convolve normally, and then
    "flip" the signal and convolve again, effectively eliminating
    the phase delay. This effectively applies the filter kernel
    twice, however. We sqrt the kernel in the frequency domain
    to somewhat compensate for this.
    '''

    yp = fft.fftshift(fft.ifftn(fft.fftn(xp) * (fft.fftn(kernelp)))).real
    yp = npj.flip(fft.fftshift(
        fft.ifftn(fft.fftn(npj.flip(yp, axis=None)) * fft.fftn(kernelp))).real,
                  axis=None)

    # remove paddings
    return _centered(yp, x_shape)


def cylindrical_filter(x, radius):
    # filter radius is in pixels, so make sure radius >= 1
    if radius < 1:
        raise ValueError("The filter radius must be at least 1 pixel")

    x = atleast_3d(x)
    xl = npj.linspace(-x.shape[0] / 2, x.shape[0] / 2, x.shape[0])
    yl = npj.linspace(-x.shape[1] / 2, x.shape[1] / 2, x.shape[1])
    zl = npj.linspace(-x.shape[2] / 2, x.shape[2] / 2, x.shape[2])

    X, Y, Z = npj.meshgrid(xl, yl, zl, sparse=True, indexing='ij')
    kernel = X**2 + Y**2 + Z**2 <= radius**2
    kernel = kernel / npj.sum(kernel.flatten())  # normalize

    return meep_filter(npj.squeeze(x), npj.squeeze(kernel))


def conic_filter(x, radius):
    x = atleast_3d(x)
    xl = npj.linspace(-x.shape[0] / 2, x.shape[0] / 2, x.shape[0])
    yl = npj.linspace(-x.shape[1] / 2, x.shape[1] / 2, x.shape[1])
    zl = npj.linspace(-x.shape[2] / 2, x.shape[2] / 2, x.shape[2])

    X, Y, Z = npj.meshgrid(xl, yl, zl, sparse=True, indexing='ij')
    kernel = np.where(
        np.abs(X**2 + Y**2 + Z**2) <= radius**2,
        (1 - np.sqrt(npj.abs(X**2 + Y**2 + Z**2)) / radius), 0)
    kernel = kernel / npj.sum(kernel.flatten())  # normalize

    return meep_filter(npj.squeeze(x), npj.squeeze(kernel))


def gaussian_filter(x, sigma):
    gaussian = lambda x, sigma: np.exp(-x**2 / sigma**2)
    x = atleast_3d(x)
    xl = npj.linspace(-x.shape[0] / 2, x.shape[0] / 2, x.shape[0])
    yl = npj.linspace(-x.shape[1] / 2, x.shape[1] / 2, x.shape[1])
    zl = npj.linspace(-x.shape[2] / 2, x.shape[2] / 2, x.shape[2])

    X, Y, Z = npj.meshgrid(xl, yl, zl, sparse=True, indexing='ij')
    kernel = np.multiply.outer(np.outer(gaussian(X, sigma), gaussian(Y,
                                                                     sigma)),
                               gaussian(Z, sigma))  # Gaussian filter kernel
    kernel = kernel / npj.sum(kernel.flatten())  # normalize

    return meep_filter(npj.squeeze(x), npj.squeeze(kernel))



'''
# ------------------------------------------------------------------------------------ #
Erosion and dilation operators
'''


def exponential_erosion(x, radius, beta):
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

    x_hat = npj.exp(beta * (1 - x))
    return 1 - npj.log(cylindrical_filter(x_hat, radius)) / beta


def exponential_dilation(x, radius, beta):
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

    x_hat = npj.exp(beta * x)
    return npj.log(cylindrical_filter(x_hat, radius)) / beta


def heaviside_erosion(x, radius, beta):
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

    x_hat = cylindrical_filter(x, radius)
    return npj.exp(-beta * (1 - x_hat)) + npj.exp(-beta) * (1 - x_hat)


def heaviside_dilation(x, radius, beta):
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

    x_hat = cylindrical_filter(x, radius)
    return 1 - npj.exp(-beta * x_hat) + npj.exp(-beta) * x_hat


def geometric_erosion(x, radius, alpha):
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
    x_hat = npj.log(x + alpha)
    return npj.exp(cylindrical_filter(x_hat, radius)) - alpha


def geometric_dilation(x, radius, alpha):
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

    x_hat = npj.log(1 - x + alpha)
    return -npj.exp(cylindrical_filter(x_hat, radius)) + alpha + 1


def harmonic_erosion(x, radius, alpha):
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
    return 1 / cylindrical_filter(x_hat, radius) - alpha


def harmonic_dilation(x, radius, alpha):
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
    return 1 - 1 / cylindrical_filter(x_hat, radius) + alpha


'''
# ------------------------------------------------------------------------------------ #
Projection filters
'''


def tanh_projection(x, beta, eta):
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

    return (npj.tanh(beta * eta) +
            npj.tanh(beta *
                     (x - eta))) / (npj.tanh(beta * eta) + npj.tanh(beta *
                                                                    (1 - eta)))


def heaviside_projection(x, beta, eta):
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

    case1 = eta * npj.exp(-beta * (eta - x) / eta) - (eta - x) * npj.exp(-beta)
    case2 = 1 - (1 - eta) * npj.exp(-beta * (x - eta) /
                                    (1 - eta)) - (eta - x) * npj.exp(-beta)
    return npj.where(x < eta, case1, case2)


'''
# ------------------------------------------------------------------------------------ #
Length scale operations
'''


def get_threshold_wang(delta, sigma):
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

    return 0.5 - special.erf(delta / sigma)


def get_eta_from_conic(b, R):
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

    norm_length = b / R
    if norm_length < 0:
        eta_e = 0
    elif norm_length < 1:
        eta_e = 0.25 * norm_length**2 + 0.5
    elif norm_length < 2:
        eta_e = -0.25 * norm_length**2 + norm_length
    else:
        eta_e = 1
    return eta_e


def get_conic_radius_from_eta_e(b, eta_e):
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
        return b / (2 * np.sqrt(eta_e - 0.5))
    elif (eta_e >= 0.75) and (eta_e <= 1):
        return b / (2 - 2 * np.sqrt(1 - eta_e))
    else:
        raise ValueError(
            "The erosion threshold point (eta_e) must be between 0.5 and 1.")


def indicator_solid(x, c, filter_f, threshold_f, resolution):
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
    gradient_filtered_field = npj.gradient(filtered_field, resolution)
    #gradient_filtered_field = gradient_filtered_field if isinstance(gradient_filtered_field,list) else [gradient_filtered_field]
    grad_mag = npj.sum(npj.array(gradient_filtered_field)**2, axis=0)
    return design_field * npj.exp(-c * grad_mag)


def constraint_solid(x, c, eta_e, filter_f, threshold_f, resolution=1):
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
    I_s = indicator_solid(x, c, filter_f, threshold_f, resolution)
    return npj.mean(I_s.flatten() *
                    npj.minimum(filtered_field.flatten() - eta_e, 0)**2)


def indicator_void(x, c, filter_f, threshold_f, resolution):
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

    filtered_field = filter_f(x)
    design_field = threshold_f(filtered_field)
    gradient_filtered_field = npj.gradient(filtered_field, resolution)
    #gradient_filtered_field = gradient_filtered_field if isinstance(list, gradient_filtered_field) else [gradient_filtered_field]
    grad_mag = npj.sum(npj.array(gradient_filtered_field)**2, axis=0)
    return (1 - design_field) * npj.exp(-c * grad_mag)


def constraint_void(x, c, eta_d, filter_f, threshold_f, resolution=1):
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
    I_v = indicator_void(x, c, filter_f, threshold_f, resolution)
    return npj.mean(I_v.flatten() *
                    npj.minimum(eta_d - filtered_field.flatten(), 0)**2)


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
    return npj.mean(4 * x.flatten() * (1 - x.flatten()))


def F_diff_open_close(x, f_open, f_close, beta=64, p=2):
    '''

    '''
    return npj.mean(
        tanh_projection(npj.abs(f_close(x) - f_open(x)).flatten()**2), beta,
        0.5) * 100


def M_diff_open_close(x, f_open, f_close, p=2):
    '''

    '''
    return npj.mean(npj.abs(f_close(x) - f_open(x)).flatten()**p) * 100


def M_diff_identity_open(x, f_open, p=2):
    '''

    '''
    return npj.mean(npj.abs(x - f_open(x)).flatten()**p) * 100


def M_diff_identity_close(x, f_close, p=2):
    '''

    '''
    return npj.mean(npj.abs(x - f_close(x)).flatten()**p) * 100
