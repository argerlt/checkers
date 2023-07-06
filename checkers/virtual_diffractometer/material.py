# -*- coding: utf-8 -*-
# This file is part of checkers, originally written by
# Austin Gerlt as part of his PhD Thesis at OSU.
#
# checkers is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License v3.
#
# Checkers is hosted at www.github.com/argerlt/checkers. For
# questions/concerns, email Austin at gerlt.1@osu.edu
"""
Created on Wed Jul  5 17:37:11 2023

@author: agerlt
"""
import numpy as np

from checkers.utils import cosd, sind, det


_FLOAT_EPS = np.finfo(float).eps

class Material():
    """ Same thing as diffpy.Structure.Lattice, but generic inputs"""

    def __init__(self, a, b, c, alpha, beta, gamma,
                 convention="a||X c*||Z", degrees=True):

        # save input values
        mags = np.array([a, b, c], dtype=np.float64)
        self.mags = mags
        angs = np.array([alpha, beta, gamma], dtype=np.float64)
        if degrees:
            angs = angs*np.pi/180
        self.angs = angs
        self.deg = angs*180/np.pi

        # calculate the convention-independent values first
        # preemtively calculate sine and cosine
        coss = np.cos(angs)
        sins = np.sin(angs)
        self.coss = coss
        self.sins = sins
        # find all the dot products
        self.dots = np.hstack([self.coss, np.array([1])])[[
            [3, 2, 1],
            [2, 3, 0],
            [1, 0, 3]]]
        # find g, the Metric Tensor (Degraef Ch 4.2)
        g = np.outer(self.mags, self.mags) * self.dots
        g[g**2 < _FLOAT_EPS] = 0
        self.g = g
        # find the inverse metric tensor
        g_star = np.linalg.inv(self.g)
        g_star[g_star**2 < _FLOAT_EPS] = 0
        self.g_star = g_star
        # find both volume and unit volume
        self.unit_vol = np.around(det(self.dots)**0.5, 12)
        self.vol = np.around(det(self.g)**0.5, 12)

        # Use the convention to calculate the direct structure matrix, a_ij.
        # this transforms from crystal coordinates to xyz.
        # This is the "A" matrix in HEXRD.
        # Write a_ij assuming p||X and s*||Z
        split_convention = [x.split("||") for x in convention.split()]
        str_2_ind = {'a': 0, 'b': 1, 'c': 2,
                     'A': 0, 'B': 1, 'C': 2,
                     'X': 0, 'Y': 1, 'Z': 2,
                     'x': 0, 'y': 1, 'Z': 2,
                     '0': 0, '1': 1, '2': 2}
        # extract the primary alignment access in direct space
        p = str_2_ind[split_convention[0][0]]
        # extract the secondary alignment access in reciprocal space
        s = str_2_ind[split_convention[1][0].split('*')[0]]
        # save out the remaining one as the "tertiary" axis
        t = list(set([0, 1, 2]) - set([p, s]))[0]
        # follow equation 7.34a from DeGraef, pre-emtively solving for "F"
        f = coss[t]*coss[s] - coss[p]
        a_ij = np.array([
            [mags[p], mags[t]*coss[s], mags[s]*coss[t]],
            [0, mags[t]*sins[s], -mags[s]*f/sins[s]],
            [0, 0, self.vol/(mags[p]*mags[t]*sins[s])]])
        # for ease of writing, the matrix above rearranges the Lattice
        # inputs based on the convention requested, essentially faking a
        # a||X c*||Z convention. Now we descramble the rows of a_ij,
        # so a/b/c/alpha/beta/gamma are back in the right order, and a_ij is
        # in the p||X s*||Z convention, where p and s are chosen by the user.
        a_ij = a_ij[[p, t, s], :]
        # finally, descramble the columns of a_ij to fully match the convention
        # chosen by the user.
        e1 = str_2_ind[split_convention[0][1]]
        e3 = str_2_ind[split_convention[1][1]]
        e2 = list(set([0, 1, 2]) - set([e1, e3]))[0]
        a_ij = a_ij[:, [e1, e2, e3]]
        # remove rouding errors and save.
        a_ij[a_ij**2 < _FLOAT_EPS] = 0
        self.a_ij = a_ij

        # use a_ij and g_star to compute b_ij. This is the "B" matrix in hexrd
        b_ij = np.dot(self.a_ij, g_star)
        b_ij[b_ij**2 < _FLOAT_EPS] = 0
        self.b_ij = b_ij

# cub = Material(3, 2, 1, 80, 60, 70)
# print(cub.a_ij)
# cub = Material(1, 2, 3, 70, 60, 80, "c||Z a*||X" )

# print(cub.a_ij)

# class Material():
#     #TODO: this should be done such that any combination of to/from primary
#     # and to/from secondary can be chosen. For now though, spoof diffpy to get
#     # HEXRD convention
#     def __init__(self, a_mag, b_mag, c_mag, alpha, beta, gamma):
#         self._a_mag = float(a_mag)
#         self._b_mag = float(b_mag)
#         self._c_mag = float(c_mag)
#         self._alpha = float(alpha)
#         self._beta = float(beta)
#         self._gamma = float(gamma)
#         self._atol = 1e-20
#         # allowed = {"a||X c*||Z": [0,0,3,3]
#         #            "c||Z a*||X": [3,3,3,3]
#         #         "c||Y b*||X",
#         #         "c||Y a*||Z",
#         #         ]
#         # self.allowed_alignments =
#         self.alignment = "a||X c*||Z"


#     @property
#     def abc(self):
#         #TODO: add reference to what this is 
#         """
#         Converts the 6 values used into define a crystal lattice into
#         three vectors defining the unit cell coordinates in crystal
#         coordinates.

#         NOTE: this follows the convention that X_crystal is parallel to a,
#         and Z_crystal is parallel to np.cross(a, b). THis is not the only
#         valid interpolation, but it is the one used for all calculations
#         in this code.
#         """
#         if hasattr(self, '_abc') is False:
#             ca = np.cos(self._alpha)
#             cb = np.cos(self._beta)
#             cg = np.cos(self._gamma)
#             sa = np.sin(self._alpha)
#             sb = np.sin(self._beta)
#             sg = np.sin(self._gamma)
#             a = self._a_mag
#             b = self._b_mag
#             c = self._c_mag
#             self._abc = np.array([
#                 [a, 0, 0],
#                 [b * cg, b * sg, 0],
#                 [c * cb, c * ca, c * sb * sa]])
#             self.abc[(self.abc**2) < self._atol] = 0
#         return(self._abc)

#     @property
#     def vol(self):
#         """Unit cell volume. equivalent to a*(b x c)"""
#         if hasattr(self, '_volume') is False:
#             abc = self.abc
#             self._volume = np.dot(abc[0], np.cross(abc[1], abc[2]))
#         return(self._volume)

#     @property
#     def abc_star(self):
#         """Reciprical lattice vectors, calculated in crystal coordinates."""
#         if hasattr(self, '_abc_star') is False:
#             abc = self.abc
#             self._abc_star = np.stack([
#                 np.cross(abc[1], abc[2]),
#                 np.cross(abc[2], abc[0]),
#                 np.cross(abc[0], abc[1])
#                 ])/self.vol
#             self._abc_star[(self._abc_star**2) < self._atol] = 0
#         return(self._abc_star)

#     @property
#     def A(self):
#         return(self.abc)

#     @property
#     def B(self):
#         return(self.abc_star)

#     @property
#     def G(self):
#         return np.dot(self.abc, self.abc)

#     def r2c(self, v_rec):
#         """
#         transforms a lattice vector in reciprical space to crystal
#         coordinates

#         Parameters
#         ----------
#         v_rec : (...,3) dimensional numpy array
#             n-dimensional numpy array of 3D vectors in reciprocal space.
#             can be any shape, but last dimension must have length 3
#         Returns
#         -------
#         v_crystal: (...,3) dimensional numpy array

#         """
#         v_rec = np.array(v_rec)
#         assert v_rec.shape[-1] == 3, "must be interpretable as an n-by-3 array"
#         # take care of trivial cases
#         if len(v_rec.shape) <= 1:
#             return np.dot(self.B, v_rec)
#         if len(v_rec.shape) == 2:
#             return(np.dot(self.B, v_rec.T)).T
#         # convert to 2d in necessary
#         n_vecs = np.prod(v_rec.shape[:-1])
#         v_rec_2d = v_rec.reshape(n_vecs, 3)
#         v_crystal_2d = np.dot(self.B, v_rec_2d.T).T
#         return(v_crystal_2d.reshape(v_rec.shape))

#     def r2l(self, v_rec, R_c2s, R_s2l, t_c2s, t_s2l):
#         """
#         transforms a lattice vector in reciprical space to lab coordinates

#         Parameters
#         ----------
#         v_rec : (...,3) dimensional numpy array
#             n-dimensional numpy array of 3D vectors in reciprocal space.
#             can be any shape, but last dimension must have length 3
#         Returns
#         -------
#         v_crystal: (...,3) dimensional numpy array

#         """
#         # initial sanity checks for vectors
#         v_rec = np.array(v_rec)
#         assert v_rec.shape[-1] == 3, "must be interpretable as an n-by-3 array"
#         # initial sanity checks for translation vectors
#         t_c2s = np.array(t_c2s, dtype=np.float64).reshape(3)
#         t_s2l = np.array(t_s2l, dtype=np.float64).reshape(3)
#         # initial sanity checks for rotation matrices
#         R_c2s = np.array(R_c2s, dtype=np.float64)
#         assert R_c2s.shape == (3, 3), "R_c2s must be 3by3 rotation matrix"
#         assert (np.linalg.det(R_c2s)-1)**2 < self._atol, ...
#         "R_c2s must be orthogonal"
#         # R_s2l checks has to  be a bit more generic, since they can change
#         # with, omega and thus we want to be able to request several at once
#         assert R_s2l.shape[-2:] == (3, 3), ...
#         "R_s2l must be interpretable as 3by3 rotation matrices"
#         assert np.mean((np.linalg.det(R_s2l)-1)**2) < self._atol, ...
#         "all R_s2l matrices must be orthogonal"

#         # pass the trivial case
#         if len(R_s2l.shape) == 2:
#             v_crystal = self._r2l_single(v_rec, R_c2s, R_s2l, t_c2s, t_s2l)
#             return v_crystal
#         # otherwise, convert R_s2l to an nx3x3 numpy array
#         n_R = np.prod(R_s2l.shape[:-2])
#         R_3D = R_s2l.reshape(n_R, 3, 3)
#         out = np.stack(
#             [self._r2l_single(v_rec, R_c2s, R, t_c2s, t_s2l) for R in R_3D]
#             )
#         # reshape and return
#         v_crystal = out.reshape(R_s2l.shape[:-2] + v_rec.shape)
#         return v_crystal

#     def _r2l_single(self, v_rec, R_c2s, R_s2l, t_c2s, t_s2l):
#         # convert v_rec to 2d array
#         if len(v_rec.shape) <= 1:
#             v_rec_2d = v_rec.reshape(1, 3)
#         if len(v_rec.shape) > 2:
#             n_vecs = np.prod(v_rec.shape[:-1])
#             v_rec_2d = v_rec.reshape(n_vecs, 3)
#         else:
#             v_rec_2d = v_rec
#         # convert to crystal space
#         v_cryst_2d = self.r2c(v_rec_2d)
#         # convert from crystal to lab (which is aligned with beam)
#         R_c2l = np.dot(R_s2l, R_c2s)
#         t_c2l = t_s2l + np.dot(R_s2l, t_c2s)
#         v_lab_2d = np.dot(R_c2l, v_cryst_2d.T).T + t_c2l
#         # reshape into original form if necessary
#         if len(v_rec.shape) <= 1:
#             return v_lab_2d.reshape(3)
#         return v_lab_2d.reshape(v_rec.shape)

#     def c2r(self, v_cryst):
#         """
#         transforms crystal coordinates into reciprocal lattice vector
#         coordinates

#         Parameters
#         ----------
#         v_cryst : (...,3) dimensional numpy array
#             n-dimensional numpy array of 3D vectors in crystal lattice
#             coordinates.
#             can be any shape, but last dimension must have length 3
#         Returns
#         -------
#         v_reciprocal: (...,3) dimensional numpy array

#         """
#         v_cryst = np.array(v_cryst)
#         assert v_cryst.shape[-1] == 3, "invalid array shape."
#         # take care of trivial cases
#         if len(v_cryst.shape) <= 1:
#             return np.dot(self.A, v_cryst)
#         if len(v_cryst.shape) == 2:
#             return(np.dot(self.B, v_cryst.T)).T
#         # convert to 2d in necessary
#         n_vecs = np.prod(v_cryst.shape[:-1])
#         v_cryst_2d = v_cryst.reshape(n_vecs, 3)
#         v_crystal_2d = np.dot(self.A, v_cryst_2d.T).T
#         return(v_crystal_2d.reshape(v_cryst.shape))

#     def l2r(self, v_lab):
#         raise NotImplementedError("""
#         Austin said there was no way you would ever need to go lab to
#         reciprocal, and he didn't write the code to do so because it looked
#         hard. If you are seeing this error, now you know who to blame.""")
#         return()



# # class Sample():
# #     def __init__():
# #         raise NotImplementedError('a')


# class Experiment():
#     def __init__():
#         xtal_locs
#         xtal_rots
#         sample_rot
#         sample_loc
        
#         raise NotImplementedError('a')


