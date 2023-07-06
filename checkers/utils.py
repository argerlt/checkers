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
Created on Wed Jul  5 17:38:10 2023

@author: agerlt
"""

import numpy as np


def cosd(ang):
    return np.cos(ang*np.pi/180)


def sind(ang):
    return np.sin(ang*np.pi/180)


def det(matrix):
    return np.linalg.det(matrix)

