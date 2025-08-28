#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np

class CurrentDensityCalculator:
    """
    Class to compute current density (jx, jy) and magnitude (j_norm) 
    from a 2D magnetic field map (B_field) using FFT and NV projection.
    
    Attributes:
        B_field (np.ndarray): 2D magnetic field array in Tesla.
        scan_size_x (float): Scan size in x-direction (meters).
        scan_size_y (float): Scan size in y-direction (meters).
        nv_angles (tuple): NV angles (phi, theta) in degrees. Default: (45, 80)
        z (float): Implantation depth in meters. Default: 20e-9
        jx, jy, j_norm (np.ndarray): Computed current density components and magnitude.
        kx, ky, k (np.ndarray): k-space vectors.
        B_field_cropped (np.ndarray): B_field after padding, cropped to original size.
    """
    
    def __init__(self, B_field: np.ndarray, scan_size_x: float, scan_size_y: float, nv_angles=(45, 80), z=20e-9):
        self.B_field = B_field
        self.scan_size_x = scan_size_x
        self.scan_size_y = scan_size_y
        self.nv_angles = nv_angles
        self.z = z
        
        self.jx = None
        self.jy = None
        self.j_norm = None
        self.kx = None
        self.ky = None
        self.k = None
        self.B_field_cropped = None

    @staticmethod
    def sph2cart(phi, theta, r):
        """Convert spherical coordinates to Cartesian."""
        rcos_theta = r * np.cos(theta)
        x = rcos_theta * np.cos(phi)
        y = rcos_theta * np.sin(phi)
        z_ = r * np.sin(theta)
        return x, y, z_

    @staticmethod
    def pad_image(image, padding_factor=1):
        """Pad image using edge padding for FFT."""
        img_size = image.shape
        y_pad = padding_factor * img_size[0]
        x_pad = padding_factor * img_size[1]
        padded = np.pad(image, mode='edge', pad_width=((y_pad//2, y_pad//2), (x_pad//2, x_pad//2)))
        return padded

    def compute(self):
        """Compute current density and store results in class attributes."""
        mu0 = 4 * np.pi * 1e-7  # Permeability of free space
        
        # Convert NV angles to radians
        phi = np.deg2rad(self.nv_angles[0])
        theta = np.deg2rad(self.nv_angles[1])
        
        # NV projection
        nx, ny, nz = self.sph2cart(phi, theta, 1)
        projection = np.array([nx, ny, nz])
        
        # Pad B_field
        padded_B = self.pad_image(self.B_field)
        
        # FFT of padded B_field
        fft_B = np.fft.fftshift(np.fft.fft2(padded_B))
        fft_B[np.isnan(fft_B) | np.isinf(fft_B)] = 0
        
        # k-space vectors
        pixel_size = self.scan_size_x / self.B_field.shape[1]  # assuming square pixels
        scaling = 2 * np.pi / pixel_size
        kx = scaling * np.fft.fftshift(np.fft.fftfreq(padded_B.shape[1]))
        ky = scaling * np.fft.fftshift(np.fft.fftfreq(padded_B.shape[0]))
        kx, ky = np.meshgrid(kx, ky)
        k = np.sqrt(kx**2 + ky**2)
        
        # Green's function
        exp_factor = np.exp(-k * self.z)
        g = -mu0 / 2 * exp_factor
        
        # Current density in k-space
        ky = -ky  # fix orientation
        jx = ky / (g * (projection[1]*ky - projection[0]*kx + 1j*projection[2]*k))
        jy = kx / (g * (projection[0]*kx - projection[1]*ky - 1j*projection[2]*k))
        
        # Hanning window filter
        hx = np.hanning(k.shape[0])
        hy = np.hanning(k.shape[1])
        img_filter = np.sqrt(np.outer(hx, hy))
        
        jx = img_filter * jx
        jy = img_filter * jy
        jx[np.isnan(jx) | np.isinf(jx)] = 0
        jy[np.isnan(jy) | np.isinf(jy)] = 0
        
        # Multiply by FFT of B-field and inverse FFT
        jx = np.fft.ifft2(np.fft.ifftshift(jx * fft_B)).real
        jy = np.fft.ifft2(np.fft.ifftshift(jy * fft_B)).real
        
        # Crop back to original size
        s0, s1 = self.B_field.shape
        start_x = (jx.shape[1] - s1) // 2
        start_y = (jx.shape[0] - s0) // 2
        self.jx = jx[start_y:start_y+s0, start_x:start_x+s1]
        self.jy = jy[start_y:start_y+s0, start_x:start_x+s1]
        self.j_norm = np.sqrt(self.jx**2 + self.jy**2)
        self.B_field_cropped = padded_B[start_y:start_y+s0, start_x:start_x+s1]
        
        # Store k-space vectors
        self.kx = kx
        self.ky = ky
        self.k = k
        
        return {
            'jx': self.jx,
            'jy': self.jy,
            'j_norm': self.j_norm,
            'kx': self.kx,
            'ky': self.ky,
            'k': self.k,
            'B_field_cropped': self.B_field_cropped
        }

