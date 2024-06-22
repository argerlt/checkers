#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 19:32:22 2024

@author: gerlt.1
"""


import numpy as np
import warnings


class Detector_Reducer():
    """
    Contains all the information on how to convert a full sized panel into
    multiple subpanel files. by default, uses data for the Dexela 2923

    Yes, this could have been a dict like Rachel did or a NamedTuple like Joel
    did, having a class makes it easier for other people to  call it with
    data from their own detectors.
    """

    def __init__(self,
                 nrows=3888,
                 ncols=3072,
                 subpanels_shape=[2, 2],
                 subpanel_names='Autogen',
                 rect=None,
                 panel_transforms=[["flip_v",], ["flip_h",]],
                 panel_names=['ff1', 'ff2']
                 ):
        self.nr = nrows
        self.nc = ncols
        self.shape = np.asanyarray(subpanels_shape).flatten()
        assert self.shape.size == 2
        if subpanel_names == 'Autogen':
            names = ["_".join([str(x), str(y)])
                     for x in range(self.shape[0])
                     for y in range(self.shape[1])
                     ]
        else:
            # if names are given, must be an iterable object (list, array, etc)
            # of type str
            names = subpanel_names
        # self.rect should be a dictonary whose keys are the names of the
        # subpanels, and whose values are in the form [[xmin,xmax][ymin,ymax]]
        # and describe the pixel locations of the subpanels relative to the
        # full panel.
        # if rect is given, it overrides subpanels_shape.
        if rect is None:
            # calc edges
            dx = np.linspace(0, self.nr, self.shape[0]+1, dtype=int)
            dy = np.linspace(0, self.nc, self.shape[1]+1, dtype=int)
            rect = dict()
            itr = 0
            for i in range(len(dx)-1):
                for j in range(len(dx)-1):
                    rect[names[itr]] = np.array(
                        [[dx[i], dx[i+1]], [dy[j], dy[j+1]]]
                        )
                    itr += 1
        # whether made or given, verify 'rect' dictionary looks right
        assert isinstance(rect, dict)
        assert np.all([isinstance(x, str) for x in rect.keys()])
        assert np.all([isinstance(x, np.ndarray) for x in rect.values()])
        self.rect = rect
        self.p_trans = panel_transforms
        if type(panel_names) == str:
            # convert naked string to single item list
            panel_names = [panel_names]
        self.p_names = panel_names

    def _parse_chore(self, chore):
        """search the chore dictionary for values, replace with defaults if
        values don't exist"""
        n = chore.get('n', 1440)
        start = chore.get('start', 0)
        stop = chore.get('stop', 360)
        skips_guess = chore.get('skips_guess', 4)
        ome_line = np.linspace(start, stop, int(n) + 1, dtype=np.float64)
        omega = np.stack([ome_line[:-1], ome_line[1:]]).T
        ome = chore.get('omega', omega)
        if not [np.min(ome), np.max(ome), len(ome)] == [start, stop, n]:
            warnings.RuntimeWarning(
                "omega array does not match the given start, stop, and " +
                "frame count. replacing values using omega array")
            n = len(ome)
        return ome, n, skips_guess


