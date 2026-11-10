"""Original study metric algorithms, retained without numerical changes."""

import math

import numpy as np
from scipy.signal import convolve2d

from ._qabf import get_Qabf
from ._ssim import ssim


def fspecial_gaussian(shape, sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',...)
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2**(4-scale+1)+1
        win = fspecial_gaussian((N, N), N/5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')
            dist = convolve2d(dist, win, mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = convolve2d(ref*ref, win, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist*dist, win, mode='valid') - mu2_sq
        sigma12 = convolve2d(ref*dist, win, mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g*sigma12

        g[sigma1_sq<1e-10] = 0
        sv_sq[sigma1_sq<1e-10] = sigma2_sq[sigma1_sq<1e-10]
        sigma1_sq[sigma1_sq<1e-10] = 0

        g[sigma2_sq<1e-10] = 0
        sv_sq[sigma2_sq<1e-10] = 0

        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=1e-10] = 1e-10

        num += np.sum(np.log10(1+g**2 * sigma1_sq/(sv_sq+sigma_nsq)))
        den += np.sum(np.log10(1+sigma1_sq/sigma_nsq))
    vifp = num/den
    return vifp

def VIF_function(A, B, F):
    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF

def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r

def SCD_function(A, B, F):
    r = corr2(F - B, A) + corr2(F - A, B)
    return r

def Qabf_function(A, B, F):
    return get_Qabf(A, B, F)

def Hab(im1, im2, gray_level):
	hang, lie = im1.shape
	count = hang * lie
	N = gray_level
	h = np.zeros((N, N))
	for i in range(hang):
		for j in range(lie):
			h[im1[i, j], im2[i, j]] = h[im1[i, j], im2[i, j]] + 1
	h = h / np.sum(h)
	im1_marg = np.sum(h, axis=0)
	im2_marg = np.sum(h, axis=1)
	H_x = 0
	H_y = 0
	for i in range(N):
		if (im1_marg[i] != 0):
			H_x = H_x + im1_marg[i] * math.log2(im1_marg[i])
	for i in range(N):
		if (im2_marg[i] != 0):
			H_x = H_x + im2_marg[i] * math.log2(im2_marg[i])
	H_xy = 0
	for i in range(N):
		for j in range(N):
			if (h[i, j] != 0):
				H_xy = H_xy + h[i, j] * math.log2(h[i, j])
	MI = H_xy - H_x - H_y
	return MI

def MI_function(A, B, F, gray_level=256):
	MIA = Hab(A, F, gray_level)
	MIB = Hab(B, F, gray_level)
	MI_results = MIA + MIB
	return MI_results

def SSIM_function(A, B, F):
    ssim_A = ssim(A, F)
    ssim_B = ssim(B, F)
    SSIM = 1 * ssim_A + 1 * ssim_B
    return SSIM.item()
