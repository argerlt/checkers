# -*- coding: utf-8 -*-
"""
Created on Tue May  9 01:17:36 2023

@author: agerlt
"""

import numpy as np
from orix.quaternion import Orientation, Symmetry, Quaternion
from typing import List, Optional, Tuple, Union




def eta_theta(
        incident_beam: Union[List, Tuple, np.ndarray],
        diffracted_beam: Union[List, Tuple, np.ndarray],
        azimuthal_zero: Optional[str] = 'east'
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the eta and theta angles relative to the original beam
    direction and the direction of the diffracted beam. Equivalent to
    equations 1-3 of Joel's diffraction guide.


    Parameters
    ----------
    incident_beam : Union[List, Tuple,np.ndarray]
        a single 3D cartesian vector describing the direction of an incoming
        beam
    diffracted_beam : Union[List, Tuple,np.ndarray]
        Either one or multiple 3D cartesian vectors describing the direction
        in which the incident_beam is being diffracted.
    azimuthal_zero:
        the direction of the zero azimuthal angle TODO: do better
    Returns
    -------
    eta: numpy array of length n
        the aximuthal angle
    theta: numpy array of length n
        the aximuthal 
    None.

    """
    # TODO: better description
    # cast i and d to numpy arrays and assert their shape
    i = np.array(incident_beam)
    d = np.array(diffracted_beam)
    assert i.shape(-1) == 3, "incident beam must be an nx3 array of vectors"
    assert d.shape(-1) == 3, "diffracted beam nust be an nx3 array of vectors"
    # normalize the vectors
    i_n = i/np.sum(i*i, axis=-1) ** 0.5
    d_n = d/np.sum(d*d, axis=-1) ** 0.5
    # convert azimuthal direction to a vector
    # TODO: this would be faster as a switch statement...
    azimuthal_dict = {'east': [1, 0, 0],
                      'west': [-1, 0, 0],
                      'north': [0, 1, 0],
                      'south': [0, -1, 0]}
    e = azimuthal_dict[str.lower(azimuthal_zero)]

    # Calc theta (ie, the angle between the two vectors)
    theta = np.arccos(np.dot(d_n, i_n))/2

    # to find the azimuthal angle, we want to find the projections of both
    # the azimuthal zero (e) and the diffraction beam (d_n) onto a plane
    # perpendicular to the beam. The pendactic way to do this is to find the
    # matrix that projects a beam onto the perpendicular plane:
    proj_perp = np.eye(3)-np.outer(i_n, i_n)
    # use it to project the incoming beam:
    d_n_perp = np.dot(proj_perp, d_n.T).T
    # then dot it with the azimuthal zero to find the angle between them
    eta = np.arccos(np.dot(e, d_n_perp.T))
    return(eta, theta)


def eta_theta_fast(
        diffracted_beam: Union[List, Tuple, np.ndarray],
        azimuthal_zero: Optional[str] = 'east'
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    same as eta_theta, but assumes e = [1,0,0] and b = [0,0,1], which makes
    math faster

    Parameters
    ----------
    diffracted_beam : Union[List, Tuple,np.ndarray]
        Either one or multiple 3D cartesian vectors describing the direction
        in which the incident_beam is being diffracted.

    Returns
    -------
    eta: numpy array of length n
        the aximuthal angle
    theta: numpy array of length n
        the aximuthal
    None.

    """
    # TODO: better description
    d = np.array(diffracted_beam)
    assert d.shape(-1) == 3, "diffracted beam nuzst be an nx3 array of vectors"
    # normalize the vectors
    d_n = d/np.sum(d*d, axis=-1) ** 0.5

    # Calc theta (ie, the angle between the two vectors)
    theta = np.arccos(d_n[2])/2
    # Calc azimuthal angle (ie, angle between the projection of the incident
    # beam onto a perpendicular plate, and the east direction on that plane.)
    eta = np.arccos(d_n[0])/2
    return(eta, theta)


