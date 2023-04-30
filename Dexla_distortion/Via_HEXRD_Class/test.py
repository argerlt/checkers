# -*- coding: utf-8 -*-
"""
Example of how to apply distortions and other stuff
"""
# imports
import numpy as np
import numba

# Hexrd imports
import hexrd
from hexrd.distortionabc import DistortionABC
from hexrd.registry import _RegisterDistortionClass
from hexrd import constants
from hexrd.constants import USE_NUMBA
from hexrd.utils import newton




# do stuff here










#%%
#########################################################
## Example of a distortion class with helpful comments ##
#########################################################
# The original version can be found at hexrd/distortion/dexly_2923.py

class Dexela_2923(DistortionABC, metaclass=_RegisterDistortionClass):
    """Joel made this Class in Jan 2021. Its a way to add sub-panel spacings
    without splitting the original ff-files (and thus doubling the
    size of your raw data.). This is more in line with HEXRD's overall philophy
    of never duplicating data unless necessary.
    
    However, it does NOT apply a per-subpanel distortion, or per-panel twist/tilt.
    Hence, the current practice is to split the 2 ff.npy files into 8 subpanel 
    files, which can be individually twisted/tiled by hexrdgui. This gives unreal
    results with overlapping panels, but it's still better than not splitting them.
    
    The Correct way to eventually apply a correction would be with a
    Distortion class similar to this one.
    
    The only required class functions are:
    
    - params
    
    - apply
    
    -apply_inverse"""

    maptype = "Dexela_2923"

    def __init__(self, params, **kwargs):
        self._params = np.asarray(params, dtype=float).flatten()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, x):
        assert len(x) == 8, "parameter list must have len of 8"
        self._params = np.asarray(x, dtype=float).flatten()

    @property
    def is_trivial(self):
        return np.all(self.params == 0)

    def apply(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_out = np.empty_like(xy_in)
            _dexela_2923_distortion(
                xy_out, xy_in, np.asarray(self.params)
            )
            return xy_out

    def apply_inverse(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_out = np.empty_like(xy_in)
            _dexela_2923_inverse_distortion(
                xy_out, xy_in, np.asarray(self.params)
            )
            return xy_out


#### private functions used by the Dexla distortion class
# I THINK Joel left these outside the function because it made numba/jit
# possible (which is WAY faster than non-numba), but IDK, this is just 
# inference as there are NO COMMENTS ANYWHERE EVER!!

# That said, I don't think you have to leave functions outside of classes to
# JIT-ify them. not sure though.
# Heres a stackoverflow on the subject:
# https://stackoverflow.com/questions/41769100/how-do-i-use-numba-on-a-member-function-of-a-class
def _find_quadrant(xy_in):
    quad_label = np.zeros(len(xy_in), dtype=int)
    in_2_or_3 = xy_in[:, 0] < 0.
    in_1_or_4 = ~in_2_or_3
    in_3_or_4 = xy_in[:, 1] < 0.
    in_1_or_2 = ~in_3_or_4
    quad_label[np.logical_and(in_1_or_4, in_1_or_2)] = 1
    quad_label[np.logical_and(in_2_or_3, in_1_or_2)] = 2
    quad_label[np.logical_and(in_2_or_3, in_3_or_4)] = 3
    quad_label[np.logical_and(in_1_or_4, in_3_or_4)] = 4
    return quad_label


@numba.njit(nogil=True, cache=True)
def _dexela_2923_distortion(out_, in_, params):
    for el in range(len(in_)):
        xi, yi = in_[el, :]
        if xi < 0.:
            if yi < 0.:
                # 3rd quadrant
                out_[el, :] = in_[el, :] + params[4:6]
            else:
                # 2nd quadrant
                out_[el, :] = in_[el, :] + params[2:4]
        else:
            if yi < 0.:
                # 4th quadrant
                out_[el, :] = in_[el, :] + params[6:8]
            else:
                # 1st quadrant
                out_[el, :] = in_[el, :] + params[0:2]

@numba.njit(nogil=True, cache=True)
def _dexela_2923_inverse_distortion(out_, in_, params):
    for el in range(len(in_)):
        xi, yi = in_[el, :]
        if xi < 0.:
            if yi < 0.:
                # 3rd quadrant
                out_[el, :] = in_[el, :] - params[4:6]
            else:
                # 2nd quadrant
                out_[el, :] = in_[el, :] - params[2:4]
        else:
            if yi < 0.:
                # 4th quadrant
                out_[el, :] = in_[el, :] - params[6:8]
            else:
                # 1st quadrant
                out_[el, :] = in_[el, :] - params[0:2]



def test_disortion():
    pts = np.random.randn(16, 2)
    qi = _find_quadrant(pts)

    # test trivial
    params = np.zeros(8)
    dc = Dexela_2923(params)
    if not np.all(dc.apply(pts) - pts == 0.):
        raise RuntimeError("distortion apply failed!")
    if not np.all(dc.apply_inverse(pts) - pts == 0.):
        raise RuntimeError("distortion apply_inverse failed!")

    # test non-trivial
    params = np.random.randn(8)
    dc = Dexela_2923(params)
    ptile = np.vstack([params.reshape(4, 2)[j - 1, :] for j in qi])
    result = dc.apply(pts) - pts
    result_inv = dc.apply_inverse(pts) - pts
    if not np.all(abs(result - ptile) <= constants.epsf):
        raise RuntimeError("distortion apply failed!")
    if not np.all(abs(result_inv + ptile) <= constants.epsf):
        raise RuntimeError("distortion apply_inverse failed!")
    return True





#%%
####################################################################
## Copy of the GE Barrel correction class, which we want to mimic ##
####################################################################
# This is a copy of the GE detector distortion class. this is the sort of thing
# we want to emulate with the dexla distortion class. I made some slight changes
# for simplicity, including:
#   - Removing the non-numba options
#
# The original version can be found at hexrd/distortion/ge_41rt.py
"""GE41RT Detector Distortion"""

RHO_MAX = 204.8  # max radius in mm for ge detector

@numba.njit(nogil=True, cache=True)
def _ge_41rt_inverse_distortion(out, in_, rhoMax, params):
    maxiter = 100
    prec = cnst.epsf

    p0, p1, p2, p3, p4, p5 = params[0:6]
    rxi = 1.0/rhoMax
    for el in range(len(in_)):
        xi, yi = in_[el, 0:2]
        ri = np.sqrt(xi*xi + yi*yi)
        if ri < cnst.sqrt_epsf:
            ri_inv = 0.0
        else:
            ri_inv = 1.0/ri
        sinni = yi*ri_inv
        cosni = xi*ri_inv
        ro = ri
        cos2ni = cosni*cosni - sinni*sinni
        sin2ni = 2*sinni*cosni
        cos4ni = cos2ni*cos2ni - sin2ni*sin2ni
        # newton solver iteration
        for i in range(maxiter):
            ratio = ri*rxi
            fx = (p0*ratio**p3*cos2ni +
                    p1*ratio**p4*cos4ni +
                    p2*ratio**p5 + 1)*ri - ro  # f(x)
            fxp = (p0*ratio**p3*cos2ni*(p3+1) +
                    p1*ratio**p4*cos4ni*(p4+1) +
                    p2*ratio**p5*(p5+1) + 1)  # f'(x)

            delta = fx/fxp
            ri = ri - delta
            # convergence check for newton
            if np.abs(delta) <= prec*np.abs(ri):
                break

        xi = ri*cosni
        yi = ri*sinni
        out[el, 0] = xi
        out[el, 1] = yi

    return out

@numba.njit(nogil=True, cache=True)
def _ge_41rt_distortion(out, in_, rhoMax, params):
    p0, p1, p2, p3, p4, p5 = params[0:6]
    rxi = 1.0/rhoMax

    for el in range(len(in_)):
        xi, yi = in_[el, 0:2]
        ri = np.sqrt(xi*xi + yi*yi)
        if ri < cnst.sqrt_epsf:
            ri_inv = 0.0
        else:
            ri_inv = 1.0/ri
        sinni = yi*ri_inv
        cosni = xi*ri_inv
        cos2ni = cosni*cosni - sinni*sinni
        sin2ni = 2*sinni*cosni
        cos4ni = cos2ni*cos2ni - sin2ni*sin2ni
        ratio = ri*rxi

        ri = (p0*ratio**p3*cos2ni
                + p1*ratio**p4*cos4ni
                + p2*ratio**p5
                + 1)*ri
        xi = ri*cosni
        yi = ri*sinni
        out[el, 0] = xi
        out[el, 1] = yi

    return out

def _rho_scl_func_inv(ri, ni, ro, rx, p):
    retval = (p[0]*(ri/rx)**p[3] * np.cos(2.0 * ni) +
              p[1]*(ri/rx)**p[4] * np.cos(4.0 * ni) +
              p[2]*(ri/rx)**p[5] + 1)*ri - ro
    return retval


def _rho_scl_dfunc_inv(ri, ni, ro, rx, p):
    retval = p[0]*(ri/rx)**p[3] * np.cos(2.0 * ni) * (p[3] + 1) + \
        p[1]*(ri/rx)**p[4] * np.cos(4.0 * ni) * (p[4] + 1) + \
        p[2]*(ri/rx)**p[5] * (p[5] + 1) + 1
    return retval


def inverse_distortion_numpy(rho0, eta0, rhoMax, params):
    return newton(rho0, _rho_scl_func_inv, _rho_scl_dfunc_inv,
                  (eta0, rho0, rhoMax, params))


class GE_41RT(DistortionABC, metaclass=_RegisterDistortionClass):

    maptype = "GE_41RT"

    def __init__(self, params, **kwargs):
        self._params = np.asarray(params, dtype=float).flatten()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, x):
        assert len(x) == 6, "parameter list must have len of 6"
        self._params = np.asarray(x, dtype=float).flatten()

    @property
    def is_trivial(self):
        return \
            self.params[0] == 0 and \
            self.params[1] == 0 and \
            self.params[2] == 0

    def apply(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_in = np.asarray(xy_in, dtype=float)
            xy_out = np.empty_like(xy_in)
            _ge_41rt_distortion(
                xy_out, xy_in, float(RHO_MAX), np.asarray(self.params)
            )
            return xy_out

    def apply_inverse(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_in = np.asarray(xy_in, dtype=float)
            xy_out = np.empty_like(xy_in)
            _ge_41rt_inverse_distortion(
                xy_out, xy_in, float(RHO_MAX), np.asarray(self.params)
            )
            return xy_out
