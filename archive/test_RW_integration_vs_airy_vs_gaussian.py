import numpy as np
from scipy.special import j0, j1, jv
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
#————— PARAMETERS ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
wavelength   = 550e-9   # incident light wavelength
NA = 0.4         # Numerical Aperture
R = 2e-6        # simulation range
#————— END ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def richards_wolf_focal_field(r, z, wavelength, NA, n=1.0, polarization='x', n_theta=300, n_phi=300):
    """
    Computes the focal electric field components using the Richards-Wolf
    vector diffraction integral.

    Based on Novotny & Hecht, Principles of Nano-Optics (2nd ed.), Ch. 3,
    and Richards & Wolf, Proc. R. Soc. London A, 253, 358-379 (1959).

    Parameters
    ----------
    r : float
        Radial position in the focal plane (meters).
    z : float
        Axial position relative to focus (meters).
    wavelength : float
        Wavelength in the medium (meters).
    NA : float
        Numerical aperture (NA = n * sin(theta_max)).
    n : float
        Refractive index of the immersion medium.
    polarization : str
        Input polarization, 'x' or 'y'.
    n_theta : int
        Number of integration points over polar angle theta.
    n_phi : int
        Number of integration points over azimuthal angle phi.

    Returns
    -------
    Ex, Ey, Ez : complex floats
        Electric field components at (r, z).
    """
    k = 2 * np.pi * n / wavelength
    theta_max = np.arcsin(NA / n)

    theta = np.linspace(0, theta_max, n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)

    dtheta = theta[1] - theta[0]
    dphi   = phi[1]   - phi[0]

    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    sin_t = np.sin(THETA)
    cos_t = np.cos(THETA)
    sin_p = np.sin(PHI)
    cos_p = np.cos(PHI)

    # aplanatic (sine condition) apodization factor: sqrt(cos theta)
    apod = np.sqrt(cos_t)

    # phase factor
    phase = np.exp(1j * k * (r * sin_t * cos_p + z * cos_t))

    # integration weight
    weight = apod * sin_t * phase * dtheta * dphi

    if polarization == 'x':
        # Novotny & Hecht eq. 3.37
        Ex_integrand = weight * (cos_t * cos_p**2 + sin_p**2)
        Ey_integrand = weight * (cos_t - 1) * sin_p * cos_p
        Ez_integrand = weight * (-sin_t * cos_p)
    elif polarization == 'y':
        Ex_integrand = weight * (cos_t - 1) * sin_p * cos_p
        Ey_integrand = weight * (cos_t * sin_p**2 + cos_p**2)
        Ez_integrand = weight * (-sin_t * sin_p)
    else:
        raise ValueError("polarization must be 'x' or 'y'")

    Ex = np.sum(Ex_integrand)
    Ey = np.sum(Ey_integrand)
    Ez = np.sum(Ez_integrand)

    return Ex, Ey, Ez


def richards_wolf_focal_plane(x, y, wavelength, NA, n=1.0, polarization='x', n_theta=100, n_phi=100):
    """
    Computes the focal field intensity map on a 2D grid at z=0.

    Parameters
    ----------
    x, y : 1D array-like
        Cartesian coordinates in the focal plane (meters).
    wavelength, NA, n, polarization, n_theta, n_phi : see richards_wolf_focal_field.

    Returns
    -------
    I : 2D np.ndarray, shape (len(x), len(y))
        Total intensity |Ex|² + |Ey|² + |Ez|².
    Ex, Ey, Ez : 2D np.ndarray
        Complex field components.
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)

    shape = R.shape
    Ex = np.zeros(shape, dtype=complex)
    Ey = np.zeros(shape, dtype=complex)
    Ez = np.zeros(shape, dtype=complex)

    for i in range(shape[0]):
        for j in range(shape[1]):
            Ex[i,j], Ey[i,j], Ez[i,j] = richards_wolf_focal_field(
                R[i,j], 0.0, wavelength, NA, n, polarization, n_theta, n_phi
            )
        # report progress every 10%
        if i % (shape[0] // 10) == 0:
            print(f"Computing focal plane: {i/shape[0]*100:.1f}%")

    I = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    return I, Ex, Ey, Ez

def airy_disk_profile(r, wavelength, NA):
    """
    Returns the normalized Airy disk intensity profile along a radial slice.

    Parameters
    ----------
    r : array-like
        Radial positions in meters (can be negative for full cross-section).
    wavelength : float
        Wavelength of light in meters.
    NA : float
        Numerical Aperture of the optical system.

    Returns
    -------
    np.ndarray
        Normalized intensity I(r) / I(0), dimensionless.
    """
    r = np.asarray(r, dtype=float)
    u = 2*np.pi*r*NA/wavelength
    intensity = np.ones_like(u)
    mask = u != 0
    intensity[mask] = (2 * j1(u[mask]) / u[mask]) ** 2
    return intensity

def gaussian(x, amp, x0, sigma, bg):
    """
    1-D Gaussian function.

    Parameters
    ----------
    x : array-like
        Independent variable.
    amp : float
        Amplitude of the Gaussian.
    x0 : float
        Center position of the Gaussian.
    sigma : float
        Standard deviation (width) of the Gaussian.
    bg : float
        Background offset.

    Returns
    -------
    np.ndarray
        Gaussian function evaluated at x.
    """
    return amp * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + bg

# make grid
x = np.linspace(-R, R, 100)  # ±10 µm
y = np.linspace(-R, R, 100)
I_rw, _,_,_ = richards_wolf_focal_plane(x, y, wavelength, NA)   # for now we only care about the intensity
I_rw = I_rw / np.max(I_rw)  # normalize to max = 1
# line cut
I_line = I_rw[:, I_rw.shape[1]//2]

# airy
I_airy = airy_disk_profile(x, wavelength, NA)

# get numerical fwhm
idmax = np.argmax(I_line)
id0 = np.argmin(np.abs(I_line[:idmax] - np.max(I_line)/2)) 
id1 = np.argmin(np.abs(I_line[idmax:] - np.max(I_line)/2)) + idmax
fwhm_RW = x[id1] - x[id0]
idmax = np.argmax(I_airy)
id0 = np.argmin(np.abs(I_airy[:idmax] - np.max(I_airy)/2)) 
id1 = np.argmin(np.abs(I_airy[idmax:] - np.max(I_airy)/2)) + idmax
fwhm_airy = x[id1] - x[id0]

# gaussian fit 
param_RW, _ = curve_fit(gaussian, x, I_line, p0=[1.0, 0.0, fwhm_RW/2.355, 0.0])
I_gaussian = gaussian(x, *param_RW)
sigma_RW = param_RW[2]
fwhm_gaussian_RW = 2.355 * sigma_RW

param_airy, _ = curve_fit(gaussian, x, I_airy, p0=[1.0, 0.0, fwhm_airy/2.355, 0.0])
I_gaussian_airy = gaussian(x, *param_airy)
sigma_airy = param_airy[2]
fwhm_gaussian_airy = 2.355 * sigma_airy

# visualize
fig, ax = plt.subplots(dpi=200)
ax.plot(x*1e6, I_line, label = "RW")
ax.plot(x*1e6, I_airy, label="Airy disk")
ax.plot(x*1e6, I_gaussian, label = f"Gaussian fit RW, fwhm = {fwhm_gaussian_RW*1e6:.3g} µm, sigma = {sigma_RW*1e6:.3g} µm", linestyle="--")
ax.plot(x*1e6, I_gaussian_airy, label = f"Gaussian fit Airy, fwhm = {fwhm_gaussian_airy*1e6:.3g} µm, sigma = {sigma_airy*1e6:.3g} µm", linestyle="--")
ax.set_xlabel("x (µm)")
ax.set_ylabel("Intensity")
ax.set_title(f"WL={wavelength*1e9:.0f} nm, NA={NA}")
ax.legend()
plt.show()