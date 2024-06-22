#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:57:06 2024

@author: gerlt.1
"""

import numpy as np
import h5py
import glob

# get chore l



class LogReader:
    """setups change based on user, beamline, and year, but in general, every
    experiment generates a spec.log file describing the experiment.
    If Your data isn't from 2022-2025 or wan't collected at ID1a3, YOU WILL
    LIKELY NEED TO WRITE YOUR OWN PARSING FUNCTION FOR THIS CLASS.
    The spec.log layout is usually as follows:
        - spec.log starts every line with a #.
        - lines that start a measurement and create a folder begin with #S
        - the number after #S is the related folder
        - the next word is the name of the macro that collected the data.
        - the numbers after the macro give the inputs. These numbers describe
            the omega start, stop, and step.
    It is theoretically possible to also reconstruct incomplete datasets on
    the fly here, as would be done if the beam was lost during a run. One
    would need to scan the self._test_data variable for "beam lost" messages
    and come up with a naming strategy for combined reconstructions. However,
    this is only necessary for nf scans, so I'm not going to bother today.
    """
    # TODO: Be a better scientist, add the on-the-fly rebuild

    def __init__(self,raw_dir, red_dir, name='spec.log', start_marker='#S'):
        self.raw_dir = raw_dir
        self.red_dir = red_dir
        # read in logged data on initialization and parse by folder.
        f_loc = os.sep.join([raw_dir,name])
        with open(f_loc, 'r') as f:
            text_data = f.read().split('\n#S ')
            f.close()
        header = text_data[0]
        self.CHESS_loc = header.split("#F ")[1].split("\n")[0]
        self.exp_name = self.CHESS_loc.split("/")[-2]
        self.collection_date = header.split("#D ")[1].split("\n")[0]
        self.spec_user = header.split(
            "#C ")[1].split("\n")[0].split("=")[1]

        self.task_list = dict()
        # some files contain no run data.
        if len(text_data) == 1:
            self.fids = np.array([], dtype=int)
            return
        self._text_data = text_data[1:]
        self._top_lines = [x.split('\n')[0] for x in text_data[1:]]
        self.fids = np.stack([int(x.split()[0]) for x in self._top_lines])
        self.macros = np.stack([x.split()[1] for x in self._top_lines])
        self.inputs = [x.split()[2:] for x in self._top_lines]

    @property
    def known_macros(self):
        """
        Dictionary of known macros. If your experiment uses one not listed
        here, write your own function, then add it to this dictionary.
        if the macro name is changed but the inputs are identical, you can
        also call the same function from different keys
        """
        return {
            'rams4_slew_ome': self._parse_rams4_slew_ome,
            'slew_ome': self._parse_rams4_slew_ome,
            'sync_ct': self._parse_sync_ct,
            'ascan': self._parse_skippable,
            'Escan': self._parse_skippable,
            'tseries': self._parse_skippable,
            }

    @property
    def _known_macro_names(self):
        return np.array([x for x in self.known_macros.keys()])

    @property
    def _num_name_dict(self):
        return {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

    def make_task_list(self, ids='all'):
        """
        clear old task_info, then run update_task_list
        """
        self.task_list = dict()
        return self.update_task_list(ids)

    def update_task_list(self, ids='all'):
        # create a list of dictionaries, each containing data necessary to
        # reduce a single experimental dataset.
        if ids == 'all':
            ids = self.fids
        # if ids aren't given as numpy array of ints, make them so
        ids = np.asanyarray(ids).astype(int)
        task_mask = np.isin(self.fids, ids)
        task_list = []
        for fid in self.fids[task_mask]:
            macro_name = self.macros[self.fids == fid][0]
            if np.isin(macro_name, self._known_macro_names):
                self.known_macros[macro_name](fid)
            else:
                print(str(fid) + " is missing a reader function. skipping...")
                print(self.raw_dir)
        return (self.task_list)

    def _parse_skippable(self, *kwargs):
        """This experiment doesnt create ff data. skip it"""
        return

    def _parse_rams4_slew_ome(self, fid):
        """rams4_slew_ome take in start,stop,count, and exposure time.
        The file usually also contains a line about number of junk images
        collected, which can change depending on omega rotation rate.
        """
        inputs = self.inputs[fid-1]
        start, stop, n, exposure = np.array(inputs[:4]).astype(float)
        ome_line = np.linspace(start, stop, int(n) + 1, dtype=np.float64)
        omega = np.stack([ome_line[:-1], ome_line[1:]]).T
        # try to find how many junk frames were saved.
        skips_str = self._text_data[fid-1].split('junk')[0].split()[-1]
        # try to parse integer, if not possible, default to 4
        skips = self._num_name_dict.get(skips_str, 4)
        d = {
            'omega': omega,
            'nframes': np.array(n, dtype=np.int64),
            'from': os.sep.join([self.raw_dir, str(fid), 'ff']),
            'to': self.red_dir,
            'skips_guess': skips,
            'exposure': exposure,
            'macro': self.macros[fid-1],
            'start': start,
            'stop': stop,
            }
        # occasionally, people will have alternate commands that effect the
        # hutch, such as running in darkfield mode. we still want to process
        # these, but also add a marker to signify they are atypical.
        if len(inputs) > 4:
            d['alt_name'] = "_".join(inputs[4:])
        self.task_list["_".join([self.exp_name, str(fid)])] = d
        return

    def _parse_sync_ct(self, fid):
        """sync_ct is a computed tomography dataset. inputs are number of
        scans, and time. The file usually also contains a line about number
        of junk images collected, which can change depending on omega
        rotation rate.
        """
        inputs = self.inputs[fid-1]
        n, exposure = np.array(inputs[:2]).astype(float)
        ome_line = np.linspace(0, 360, int(n) + 1, dtype=np.float64)
        omega = np.stack([ome_line[1:], ome_line[:-1]]).T
        # try to find how many junk frames were saved.
        skips_str = self._text_data[fid-1].split('junk')[0].split()[-1]
        # try to parse integer, if not possible, default to 4
        skips = self._num_name_dict.get(skips_str, 4)
        d = {
            'omega': omega,
            'nframes': np.array(n, dtype=np.int64),
            'from': os.sep.join([self.raw_dir, str(fid), 'ff']),
            'to': self.red_dir,
            'skips_guess': skips,
            'exposure': exposure,
            'macro': self.macros[fid-1],
            }
        # occasionally, people will have alternate commands that effect the
        # hutch, such as running in darkfield mode. we still want to process
        # these, but also add a marker to signify they are atypical.
        if len(inputs) > 4:
            d['alt_name'] = "_".join(inputs[4:])
        self.task_list["_".join([self.exp_name, str(fid)])] = d
        return