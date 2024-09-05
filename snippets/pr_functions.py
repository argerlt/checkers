#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:58:10 2024

@author: gerlt.1
"""

import cc3d
import cv2
import glob
import h5py
import os
import sparse
import warnings

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from hexrd.utils.hdf5 import unwrap_dict_to_h5
import numpy as np
import scipy.stats as stats


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

    def __init__(self, raw_dir, red_dir, name='spec.log',
                 start_marker='#S', nf=False):
        self.raw_dir = raw_dir
        self.red_dir = red_dir
        # read in logged data on initialization and parse by folder.
        f_loc = os.sep.join([raw_dir, name])
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
        # include flag if nf (and therefore, tif images.)
        self.nf = nf
        self.make_task_list()

    @property
    def known_macros(self):
        """
        Dictionary of known macros. If your experiment uses one not listed
        here, write your own function, then add it to this dictionary.
        if the macro name is changed but the inputs are identical, you can
        also call the same function from different keys
        """
        macros = {
            'rams4_slew_ome': self._parse_rams4_slew_ome_ff,
            'slew_ome': self._parse_rams4_slew_ome_ff,
            'sync_ct': self._parse_sync_ct,
            'ascan': self._parse_skippable,
            'Escan': self._parse_skippable,
            'tseries': self._parse_skippable,
            }

        nf_macros = {
            'rams4_slew_ome': self._parse_rams4_slew_ome_nf,
            'slew_ome': self._parse_rams4_slew_ome_nf,
            }
        if self.nf:
            macros.update(nf_macros)
        return macros

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
        # task_list = []
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

    def _parse_rams4_slew_ome_nf(self, fid):
        """rams4_slew_ome take in start,stop,count, and exposure time.
        The file usually also contains a line about number of junk images
        collected, which can change depending on omega rotation rate.
        """
        dict_name = "_".join([self.exp_name, str(fid)])
        txt = self._text_data[fid-1]
        # before starting, check to see if experiment was automatically
        # repeated. this can happen due to beam loss, detector error, etc.
        if "repeating scan" in txt.lower():
            print("{} was repeated. skipping...".format(dict_name))
            return
        # assuming no skip, find the inputs, use them to generate omega data.
        inputs = self.inputs[fid-1]
        start, stop, n, exposure = np.array(inputs[:4]).astype(float)
        ome_line = np.linspace(start, stop, int(n) + 1, dtype=np.float64)
        omega = np.stack([ome_line[:-1], ome_line[1:]]).T

        # NF saves tiffs, with before AND after empties. Grab start/stop
        # TIFF names
        img_init = txt.split("NF saving")[1].split(
            ".tif.\n")[0].split("nf_")[-1]
        # occasionally, beam loss and/or manual intervention causes
        # a restart and partial redo. Asuume what WAS collected is still
        # important, but modify the nframes and omega accordingly.
        restarted = False
        if "repeating the current slew_scan..." in txt.lower():
            print("{} was automatically restarted.".format(dict_name))
            restarted = True
        elif "aborted" in txt.lower():
            print("{} was restarted by user.".format(dict_name))
            restarted = True
        if restarted:
            img_start = str(int(img_init)+4).zfill(6)
            img_stop = txt.split("NF saved last")[1].split(
                "nf_")[1].split(".tif")[0]
            # also update the omega and nframes accordingly
            n = int(img_stop)-int(img_start)
            omega = omega[:n]
            stop = np.max(omega)
        else:
            # NF saves medians before and after, and saves as TIF. lets
            # explicitly grab some filenames.
            img_start = txt.split("NF first")[1].split(
                ".tif.\n")[0].split("nf_")[-1]
            img_stop = txt.split("NF last")[1].split(
                ".tif.\n")[0].split("nf_")[-1]
        skips = int(img_start) - int(img_init)

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
            'img_start': img_start,
            'img_stop': img_stop,
            }
        # occasionally, people will have alternate commands that effect the
        # hutch, such as running in darkfield mode. we still want to process
        # these, but also add a marker to signify they are atypical.
        if len(inputs) > 4:
            d['alt_name'] = "_".join(inputs[4:])
        self.task_list[dict_name] = d
        return

    def _parse_rams4_slew_ome_ff(self, fid):
        """rams4_slew_ome take in start,stop,count, and exposure time.
        The file usually also contains a line about number of junk images
        collected, which can change depending on omega rotation rate.
        """
        dict_name = "_".join([self.exp_name, str(fid)])
        txt = self._text_data[fid-1]
        # before starting, check to see if experiment was automatically
        # repeated. this can happen due to beam loss, detector error, etc.
        if "repeating scan" in txt.lower():
            print("{} was repeated. skipping...".format(dict_name))
            return
        # assuming no skip, find the inputs, use them to generate omega data.
        inputs = self.inputs[fid-1]
        start, stop, n, exposure = np.array(inputs[:4]).astype(float)
        ome_line = np.linspace(start, stop, int(n) + 1, dtype=np.float64)
        omega = np.stack([ome_line[:-1], ome_line[1:]]).T
        # try to find how many junk frames were saved.
        skips_str = txt.split('junk')[0].split()[-1]
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
        self.task_list[dict_name] = d
        return

    def _parse_sync_ct(self, fid):
        """sync_ct is a computed tomography dataset. inputs are number of
        scans, and time. The file usually also contains a line about number
        of junk images collected, which can change depending on omega
        rotation rate.
        """
        dict_name = "_".join([self.exp_name, str(fid)])
        txt = self._text_data[fid-1]
        # before starting, check to see if experiment was automatically
        # repeated. this can happen due to beam loss, detector error, etc.
        if "repeating scan" in txt.lower():
            print("{} was repeated. skipping...".format(dict_name))
            return
        # assuming no skip, find the inputs, use them to generate omega data.
        inputs = self.inputs[fid-1]
        n, exposure = np.array(inputs[:2]).astype(float)
        ome_line = np.linspace(0, 360, int(n) + 1, dtype=np.float64)
        omega = np.stack([ome_line[1:], ome_line[:-1]]).T
        # try to find how many junk frames were saved.
        skips_str = txt.split('junk')[0].split()[-1]
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
        self.task_list[dict_name] = d
        return


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


# NOTE TO FUTURE ME: the unique call in this function is well over half the
# total time of this process, and I think numba/jit would speed it up, but
# I couldn't get it to work as I wanted. cc3d must be put into it's own
# function first (makes sense, it's a cpp wrapper) but even after, np.unique
# throws a fit because I use "too many positional arguments", which i think
# means jit doesn't have a function for returning indexes from unique (which
# i need)
def cc3d_ff_feature_finder(data_slice,
                           smallest_free_fid,
                           k,
                           lowest_observed_layer=0,
                           min_spot_size=30,
                           min_total_spot_intensity=2000,
                           ):
    # get binarized yes/no, use it to assign spot IDs.
    binarized = (data_slice > 0).astype(np.int8)
    feature_map = cc3d.connected_components(binarized)
    # do a per-layer unique search to throw out spots smaller than 2x2. These
    # are nearly always burned out pixels that connect spots over multiple
    # omegas.
    l_count = data_slice.shape[0]
    for i in range(l_count):
        # toss 3x3 burned out pixels connecting spots accross omegas.
        f2v = np.where([feature_map[i] > 0])
        if f2v[0].size < 1:
            continue
        fm_flat = feature_map[i][f2v[1:]]
        fid, inv, count = np.unique(fm_flat, False, True, True)
        fid[count < 10] = 0
        fid[fid > 0] = 1
        fid = fid.astype(np.int8)
        binarized[i][f2v[1:]] = fid[inv]
    del fid, inv, count, i
    # redo the feature map.
    feature_map = cc3d.connected_components(binarized)

    # Get the Feature ID's of spots that go into the next unlooked at area.
    # we will need to re-run any layers containing these unfinished grains.
    unfinished_spots = np.unique(feature_map[-1])
    # similarly, document spots in the bottom layer, as these should already
    # be documented from previous searches
    old_spots = np.unique(feature_map[0])

    # at This point, 99(ish)% of the feature map is zeros, so to make the
    # unique search faster, lets get a vector of just the non-zero parts
    f2v = np.where(feature_map[:-1] > 0)
    # do a unique search on just the non-zero data
    fids, idxs, inv, counts = np.unique(feature_map[f2v], True, True, True)
    # find the lowest layer with an unfinished spots, but ignore stringers
    # that run through EVERY layer
    redo_layers = f2v[0][idxs[np.isin(fids, unfinished_spots)]]
    if redo_layers.size == 0:
        redo_count = 0
    else:
        not_bottom_redos = redo_layers[redo_layers > 0]
        if not_bottom_redos.size > 0:
            redo_count = l_count - np.min(redo_layers[redo_layers > 1])
        else:
            redo_count = 0

    # Now cleanup remaining feature ids.
    # find spots that are big, complete, and new
    big_spots = fids[counts > min_spot_size]
    finished_spots = big_spots[~np.isin(big_spots, unfinished_spots)]
    finished_spots = big_spots[~np.isin(big_spots, old_spots)]
    # make one final binarized map and feature map, just so our feature IDs
    # are sequential. this is unnecessary, but handy.
    binarized = binarized*0
    binarized[np.isin(feature_map, finished_spots)] = 1
    feature_map = cc3d.connected_components(binarized)
    final_spot_ids = np.unique(feature_map[feature_map > 0])
    # cleanup, bc memory is precious
    del inv, counts, idxs, fids, binarized, f2v, old_spots
    del redo_layers, unfinished_spots

    # done with dense data, so lets switch to sparse matrices.
    sparse_fm = sparse.COO(feature_map)
    del feature_map
    spots = np.zeros([len(final_spot_ids), 4])
    spot_ids = np.zeros(len(final_spot_ids), dtype=np.uint32)

    # calculate per-spot data
    spot_id = 0
    for fid in final_spot_ids:
        loc = np.where(sparse_fm == fid)
        if loc[0].max() <= lowest_observed_layer:
            continue  # This spot was already fully observed. Ignore.
        val = data_slice[loc]
        val_sum = np.sum(val)
        if val_sum < min_total_spot_intensity:
            sparse_fm.data[sparse_fm.data == fid] = 0
            continue  # Triggers if too small. discards and moves on.
        spots[spot_id, 0] = val_sum
        spots[spot_id, 1:] = np.average(np.stack(loc), weights=val, axis=1)
        spot_ids[spot_id] = fid
        spot_id += 1

    # now that all the cleaning is done, grab the original pixel intensities.
    panel_vals = data_slice[sparse_fm.coords[0],
                            sparse_fm.coords[1],
                            sparse_fm.coords[2],
                            ]
    # cleanup to help with memory leak
    del data_slice

    # now that all the big datasets are explicitly deleted, allow the code
    # to exit if it found no spots whatsoever
    if spot_id < 1:  # triggers if no spots were found.
        del sparse_fm
        d1 = np.zeros([0, 4], dtype=float)
        d2 = np.zeros([0, 3], dtype=np.int16)
        d3 = np.zeros([0, 1], dtype=float)
        d4 = np.zeros([0, 1], dtype=int)
        return d1, d2, d3, d4, redo_count, k

    # If spots were found, cut off unused entrys in list
    spot_ids = spot_ids[:spot_id]
    spots = spots[:spot_id, :]

    # create translator for reassigning spots to keep with sequential ids
    replacer = np.zeros(sparse_fm.data.max()+1, dtype=int)
    replacer[spot_ids] = np.arange(smallest_free_fid,
                                   smallest_free_fid + spot_id)
    final_fids = replacer[sparse_fm.data]

    # throw out the sparse entrys that were set to zero
    mask = final_fids > 0
    crds = (sparse_fm.coords.T[mask]).astype(np.int16)
    vals = np.atleast_2d(panel_vals[mask]).T
    fids = np.atleast_2d(final_fids[mask]).T
    del sparse_fm
    return spots, crds, vals, fids, redo_count, k


def spoof_frame_cache(name, coords, vals, subpanel_shape):
    arrd = {}
    for i in np.arange(subpanel_shape[0]):
        mask = coords[:, 0] == i
        rc = coords[mask]
        d = (vals[mask]).flatten()
        arrd[f'{i}_row'] = rc[:, 1]
        arrd[f'{i}_col'] = rc[:, 2]
        arrd[f'{i}_data'] = d
    arrd['shape'] = subpanel_shape[1:]
    arrd['nframes'] = subpanel_shape[0]
    arrd['dtype'] = str(d.dtype).encode()
    np.savez_compressed(name, **arrd)
    short_name = name.split(os.sep)[-1]
    return short_name


def spoof_frame_cache_w_fid(name, coords, vals, fids, subpanel_shape):
    arrd = {}
    for i in np.arange(subpanel_shape[0]):
        mask = coords[:, 0] == i
        rc = coords[mask]
        d = (vals[mask]).flatten()
        f = (fids[mask]).flatten()
        arrd[f'{i}_row'] = rc[:, 1]
        arrd[f'{i}_col'] = rc[:, 2]
        arrd[f'{i}_data'] = d
        arrd[f'{i}_fid'] = f
    arrd['shape'] = subpanel_shape[1:]
    arrd['nframes'] = subpanel_shape[0]
    arrd['dtype'] = str(d.dtype).encode()
    np.savez_compressed(name, **arrd)
    short_name = name.split(os.sep)[-1]
    return short_name


def reduce_entire_ff(chore, chore_name, det_red, instr_dict, n_med_frames=20,
                     create_npz=False, spoof_thresholded=False,
                     save_threshold=250):
    mp_id = "p_"+str(os.getpid() % 41)
    a = " ==================================== \n"
    b = "    {}: starting {}\n".format(mp_id, chore_name)
    print(a + b + a)
    # REMINDER!!!!!! In the id1a3 setup, ff1 is on the RIGHT, and ff2
    # is on the LEFT (opposite of assumption)
    f_h5L = h5py.File(glob.glob(chore['from']+os.sep+"*ff2*")[0], 'r')
    f_h5R = h5py.File(glob.glob(chore['from']+os.sep+"*ff1*")[0], 'r')
    dat_h5L = f_h5L['imageseries/images']
    dat_h5R = f_h5R['imageseries/images']

    # because it's annoying to finish all the processing, only to fail during
    # saving, pre-flight the hdf5 save file.
    first_part = chore['to'] + os.sep + chore_name
    h5_save_name_full = first_part + '.sparse'
    h5_save = h5py.File(h5_save_name_full, 'w')
    for thing in ['load', 'epoch', 'z_height', 'nframes',
                  'from', 'start', 'stop', 'exposure']:
        h5_save.attrs[thing] = chore[thing]
    settings_grp = h5_save.create_group('settings/initial_instr')
    unwrap_dict_to_h5(settings_grp, instr_dict)

    # begin loading frames for median filter
    n_frames, n_skips = det_red._parse_chore(chore)[1:]
    med_skip = np.floor(n_frames/n_med_frames).astype(int)
    # first step is to find the per-pixel medians over the whole omega
    med_L = np.median(dat_h5L[4::med_skip], axis=0).astype(np.uint16)
    med_R = np.median(dat_h5R[4::med_skip], axis=0).astype(np.uint16)
    # next, assume background noise follows Poisson distribution
    # (reasonable assumption, as per Ralph's MTEX paper). calc the 99.99%
    # threshold.
    # This is the threshold where, after subtracting the per-pixel variance,
    # we can be 99.99% sure data above this intensity is NOT purely background
    bg = (np.mean(med_L) + np.mean(med_R))/2
    bg_thresh = stats.poisson(bg).ppf(0.9999) - bg

    # Set up per-subpanel containers for holding data
    panel_shape = np.array(dat_h5L.shape) - (n_skips, 0, 0)
    subpanel_shape = panel_shape//[1, det_red.shape[0], det_red.shape[1]]
    per_panel_keys = det_red.rect.keys()
    spnl_keys = [a + b for a in ['ff1_', 'ff2_'] for b in per_panel_keys]
    data = dict([(k, np.zeros(subpanel_shape + (11, 0, 0), dtype=np.int16))
                 for k in spnl_keys])
    spots = dict([(k, np.zeros([0, 4], dtype=float)) for k in spnl_keys])
    coords = dict([(k, np.zeros([0, 3], dtype=np.int16)) for k in spnl_keys])
    vals = dict([(k, np.zeros([0, 1], dtype=float)) for k in spnl_keys])
    fids = dict([(k, np.zeros([1, 1], dtype=int)) for k in spnl_keys])

    # set up trackers for which layers have been loaded, discarded, and
    # previously scanned.
    l_start = n_skips
    # the layerid corresponding to the data in data[k][0,:,:]
    min_layer_in_spnl = dict([(k, n_skips-1) for k in spnl_keys])

    while l_start < n_frames:
        # we want to load frames until one of 3 things happen:
        #   1) the subpanel with the fewest frames has 40 total
        #   2) the subpanel with the most frames has 80 toal
        #   3) we load in the last frame.
        # find how many layers this requires and load that many. If that
        # number is less than 5, load 5 instead.
        # NOTE TO FUTURE AUSTIN: In a previous version of this, you had the
        # code load 8 frames at a time, so you could asyncronously load when
        # multiprocessing. this caused all sorts of bookeeping problems.
        # just loading full chuncks as you need them is simpler, if slower,
        # but on OSC, it makes zero difference b/c we aren't parallelizing
        # loads per-node due to memory constraints.
        spnl_max = max(min_layer_in_spnl.values())
        spnl_min = min(min_layer_in_spnl.values())
        if spnl_max-spnl_min < 40:
            # delta between the stacks is small, fill smallest to 40
            l_stop = np.max([spnl_max + 40, l_start+5])
        else:
            # delta between stacks is big, fill biggest to 80
            l_stop = np.max([spnl_min + 80, l_start+5])
        # if l_stop is bigger than n_frames-5, just load everything left
        if l_stop > (n_frames-5):
            l_stop = n_frames + n_skips
        l_delta = l_stop-l_start
        # load new data
        # REMINDER! This line adds the horizontal flip to ff1 (Right panel)
        # and the vertical flip to ff2 (Left panel)
        # AS A NOTE THOUGH, the origin for images is in the TOP LEFT, not
        # the bottom left. This is corrected for when converting between
        # sparse representation and lab_xyz (not part of this function).
        # AUSTIN! you lost all of 8/29/2024 to not documenting this flip, and
        # also writing it backwards. clean your code, dummy.
        dataL = (np.asanyarray(
            dat_h5L[l_start:l_stop], dtype=np.int16) - med_L)[:, ::-1, :]
        dataR = (np.asanyarray(
            dat_h5R[l_start:l_stop], dtype=np.int16) - med_R)[:, :, ::-1]
        dataL[dataL < bg_thresh] = 0
        dataR[dataR < bg_thresh] = 0
        # append it to data stacks
        for ppk in per_panel_keys:
            [[a, b], [c, d]] = det_red.rect[ppk]
            kr, kl = 'ff1_' + ppk, 'ff2_' + ppk
            spnl_startr = l_start - min_layer_in_spnl[kr]
            spnl_startl = l_start - min_layer_in_spnl[kl]
            data[kr][spnl_startr:spnl_startr + l_delta] = dataR[:, a:b, c:d]
            data[kl][spnl_startl:spnl_startl + l_delta] = dataL[:, a:b, c:d]
        if l_stop < (n_frames+n_skips):
            # get number of explored layers. if a subpanel doesn't have
            # at least 20 unexplored panels loaded, it's not worth reducing,
            # so skip it for now
            reducable = [k for k in spnl_keys if
                         l_stop - min_layer_in_spnl[k] > 20]
            if len(reducable) < 1:
                # shouldn't be possible to have NO reducible layers, but if we
                # do, just try agian.
                print("wat?")
                l_start = l_stop + 0
                continue
        else:
            # this line tricks the reducer into thinking there is one
            # extra loaded layer of all zeros, thus forcing the spot finder
            # to terminate it's search.
            l_stop = l_stop + 1
            reducable = [k for k in spnl_keys]

        # start parallel thread executor
        executor = ProcessPoolExecutor(8)
        # print("{}: -- running reduction ... ".format(mp_id))
        print("{}: - layers {} to {} loaded, starting reduction".format(
            mp_id, l_start, l_stop))
        # series test line. leave commented during run
        # k = spnl_keys[0]
        # a=cc3d_ff_feature_finder(data[k][:(l_stop-min_layer_in_spnl[k])],1,k)
        futures = {executor.submit(
            cc3d_ff_feature_finder,
            data[k][:(l_stop - min_layer_in_spnl[k])],
            np.max(fids[k]+1),
            k,
            l_start - min_layer_in_spnl[k]):
                k for k in reducable}
        # REAL FAST!: h5py locks the GIL, so it can't pre-load data in the
        # background. However, for the next 3-ish seconds, we are waiting on
        # subprocesses, so we can preload some data from the h5py, which is
        # the bottleneck of this process. It doesn't matter that we don't
        # use this data, the important part is h5py caches these layers.
        if l_stop < n_frames:
            cache = dat_h5L[l_stop:np.min(l_stop+10, n_frames)]
            cache = dat_h5R[l_stop:np.min(l_stop+10, n_frames)]
        for future in as_completed(futures):
            out = future.result()
            futures.pop(future)  # stop memory leak
            k = out[-1]
            old_l = min_layer_in_spnl[k]
            # up until here, data has been in (omega,x,y) format, as it's
            # easier to load that way, but it makes more sense to save the
            # sparsified final data (x,y,omega).
            new_spots = out[0][:, (0, 2, 3, 1)] + [0, 0, 0, old_l - n_skips]
            new_coords = out[1][:, (1, 2, 0)] + [0, 0, old_l - n_skips]
            spots[k] = np.vstack([spots[k], new_spots])
            coords[k] = np.vstack([coords[k], new_coords])
            vals[k] = np.vstack([vals[k], out[2]])
            fids[k] = np.vstack([fids[k], out[3]])
            removable_layers = l_stop - min_layer_in_spnl[k] - out[4]
            data[k] = data[k][removable_layers:]
            min_layer_in_spnl[k] = l_stop - out[4]
        executor.shutdown()

        # update on how many layers were purged
        purged = max(min_layer_in_spnl.values()) - spnl_max
        print("{}: -- {} layers done and purged ".format(mp_id, purged))
        # reset l_start for next loop
        l_start = l_stop + 0
        if l_start > n_frames:
            if l_start == n_frames+n_skips+1:
                print("{}: -- reduction completed".format(mp_id))
            else:
                print("XXXXX WARNING: I think I made a goof XXXXX")

    # IF you made it this far, congrats, reduction is done.
    # save to h5py
    print("{}: +++ saving h5 to {}".format(mp_id, chore['to']))
    for k in spnl_keys:
        # all
        grp = h5_save.create_group('data/' + k)
        grp.create_dataset('spots', data=spots[k], compression='gzip')
        grp.create_dataset('coords', data=coords[k], compression='gzip')
        grp.create_dataset('vals', data=vals[k], compression='gzip')
        grp.create_dataset('fids', data=fids[k][1:], compression='gzip')
    h5_save.close()

    # save aggressively thresholded h5py if requested
    if spoof_thresholded:
        h5_save_name_sml = first_part + 'T{}.sparser'.format(save_threshold)
        h5_save_sml = h5py.File(h5_save_name_sml, 'w')
        masks = dict([(k, vals[k] > save_threshold) for k in spnl_keys])
        m_coords = dict(
            [(k, coords[k][masks[k].flatten(), :]) for k in spnl_keys])
        m_vals = dict([(k, vals[k][masks[k]]) for k in spnl_keys])
        m_fids = dict([(k, fids[k][1:][masks[k]]) for k in spnl_keys])
        for k in spnl_keys:
            gs = h5_save_sml.create_group('data/' + k)
            gs.create_dataset('spots', data=spots[k], compression='gzip')
            gs.create_dataset('coords', data=m_coords[k], compression='gzip')
            gs.create_dataset('vals', data=m_vals[k], compression='gzip')
            gs.create_dataset('fids', data=m_fids[k][1:], compression='gzip')
        for thing in ['load', 'epoch', 'z_height', 'nframes',
                      'from', 'start', 'stop', 'exposure']:
            h5_save_sml.attrs[thing] = chore[thing]
        settings_grp = h5_save_sml.create_group('settings/initial_instr')
        unwrap_dict_to_h5(settings_grp, instr_dict)
        h5_save_sml.close()

    # also spoof framecaches if requested
    if create_npz:
        executor = ThreadPoolExecutor(len(spnl_keys)*2)
        futures = {executor.submit(
            spoof_frame_cache,
            first_part+"-" + k + ".npz",
            coords[k],
            vals[k],
            subpanel_shape):
                k for k in spnl_keys}
        fid_futures = {executor.submit(
            spoof_frame_cache_w_fid,
            first_part+"-fid-" + k + ".npz",
            coords[k],
            vals[k],
            fids[k][1:],
            subpanel_shape):
                k for k in spnl_keys}
        futures.update(fid_futures)
        if spoof_thresholded:
            sparse_futures = {executor.submit(
                spoof_frame_cache,
                first_part+"-" + k + "T{}.npz".format(save_threshold),
                m_coords[k],
                m_vals[k],
                subpanel_shape):
                    k for k in spnl_keys}
            futures.update(sparse_futures)
            sparse_fid_futures = {executor.submit(
                spoof_frame_cache_w_fid,
                first_part+"-fid-" + k + "T{}.npz".format(save_threshold),
                m_coords[k],
                m_vals[k],
                m_fids[k],
                subpanel_shape):
                    k for k in spnl_keys}
            futures.update(sparse_fid_futures)
        for future in as_completed(futures):
            out = future.result()
            futures.pop(future)  # stop memory leak
            print("saved " + future.result())
        executor.shutdown()

    # cleanup to stop mem leak when multithreading
    del coords, vals, spots, fids, dataL, dataR
    del executor, data
    return


def cc3d_nf_feature_finder(data_slice,
                           smallest_free_fid,
                           min_spot_size=250,
                           min_total_spot_intensity=100000,
                           ):
    # get binarized yes/no, use it to assign spot IDs.
    binarized = (data_slice > 0).astype(np.int8)
    feature_map = cc3d.connected_components(binarized)
    del binarized
    # get the id's of the spots that go into the next unlooked at area
    unfinished_spots, ufc = np.unique(feature_map[-1], False, False, True)
    # get the id's of already documented spots
    old_spots = np.unique(feature_map[0])
    # toss out ones with fewer than 30 pixels of data (almost always noise)
    unfinished_spots = unfinished_spots[ufc > 30]
    # clump the feature maps by id.
    fids, idxs, inv, counts = np.unique(feature_map[:-1], True, True, True)

    # find first layer with an unfinished grain in it
    unfinished_fids = idxs[np.isin(fids, unfinished_spots[1:])]
    # there are spots on the camera that flicker on/of for multiple layers
    # and appear as multi-panel stringers. dump them.
    pix_per_layer = feature_map[0].size
    x = np.max([2, feature_map.shape[0] - 8])
    unfinished_fids = unfinished_fids[unfinished_fids > pix_per_layer * x]
    # floor divide to get highest layer with no incomplete spots
    if unfinished_fids.size == 0:
        redo_count = 0
    else:
        bot_redo_layer = np.min(unfinished_fids // pix_per_layer)
        redo_count = data_slice.shape[0] - bot_redo_layer

    # find big spots that are complete, new , and not background
    big_spots = fids[counts > min_spot_size]
    finished_spots = big_spots[~np.isin(big_spots, unfinished_spots)]
    finished_spots = big_spots[~np.isin(big_spots, old_spots)]
    finished_spots = finished_spots[finished_spots > 0]

    # use the "good" spots to clean the feature map
    # NOTE: this is replacing the "bads" with zeros, so when I invert, the
    # old features are assigned zeros (background) instead of their old fid
    cleaned_fids = (fids*np.isin(fids, finished_spots))
    cleaned_fm = np.reshape(cleaned_fids[inv], feature_map[:-1].shape)
    del feature_map, inv, counts, idxs, fids

    # sparsify to save time/space, then find the size and center of each spot
    sparse_fm = sparse.COO(cleaned_fm)
    spots = np.zeros([len(finished_spots), 4])
    spot_ids = np.zeros(len(finished_spots), dtype=np.uint32)
    spot_id = 0
    for fid in finished_spots:
        loc = np.where(sparse_fm == fid)
        # if loc[0].max() < explored_layers:
        #     continue  # triggers if spot was already previously observed
        val = data_slice[loc]
        val_sum = np.sum(val)
        if val_sum < min_total_spot_intensity:
            sparse_fm.data[sparse_fm.data == fid] = 0
            continue  # Triggers if too small. discards and movees on.
        spots[spot_id, 0] = val_sum
        spots[spot_id, 1:] = np.average(np.stack(loc), weights=val, axis=1)
        spot_ids[spot_id] = fid
        spot_id += 1
    # cleanup to help with memory leak
    panel_vals = data_slice[sparse_fm.coords[0],
                            sparse_fm.coords[1],
                            sparse_fm.coords[2],
                            ]
    del cleaned_fm, cleaned_fids, finished_spots, big_spots, old_spots
    del data_slice, ufc, unfinished_spots, unfinished_fids
    if spot_id < 1:  # triggers if no spots were found
        del sparse_fm
        d1 = np.zeros([0, 4], dtype=float)
        d2 = np.zeros([0, 3], dtype=np.int16)
        d3 = np.zeros([0, 1], dtype=float)
        d4 = np.zeros([0, 1], dtype=int)
        return d1, d2, d3, d4, redo_count
    # cut off unused entrys in list
    spot_ids = spot_ids[:spot_id]
    spots = spots[:spot_id, :]

    # create translator for reassigning spots to keep with sequential ids
    replacer = np.zeros(sparse_fm.data.max()+1, dtype=int)
    replacer[spot_ids] = np.arange(smallest_free_fid,
                                   smallest_free_fid + spot_id)
    final_fids = replacer[sparse_fm.data]

    # throw out the sparse entrys that were set to zero
    mask = final_fids > 0
    crds = (sparse_fm.coords.T[mask]).astype(np.int16)
    vals = np.atleast_2d(panel_vals[mask]).T
    fids = np.atleast_2d(final_fids[mask]).T
    del sparse_fm
    return spots, crds, vals, fids, redo_count


# f_1 = h5py.File("red/me3-6-broke-nf-1_1.sparse", 'r')
# crds = np.array(f_1['coords'])
# dims = f_1.attrs['xyo_dims']
# vals = np.array(f_1['vals'])
# f_1.close()

# spoof_frame_cache('test', crds, vals, dims)

# %%


def reduce_entire_nf(chores, chore_name, n_med_frames=30, thresh=20, mpid=0):
    chore = chores[chore_name]
    if mpid == 0:
        mpid = os.getpid() % 6
    mp_id = "p_"+str(mpid)
    a = " ==================================== \n"
    b = "    {}: starting {}\n".format(mp_id, chore_name)
    print(a + b + a)
    # get files, sort them, throw out start/stops
    all_tifs = glob.glob(chore['from'][:-2]+"nf"+os.sep+("*.tif"))
    all_tifs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    n_frames = np.atleast_1d(chore['nframes'])[0]
    n_skips = np.atleast_1d(chore['skips_guess'])[0]
    tifs = np.array(all_tifs)[n_skips:n_frames+n_skips]

    # get median to subtract
    med_skip = np.floor(n_frames/n_med_frames).astype(int)
    med_tifs = tifs[med_skip::med_skip]
    # for some *** reason, med tiffs comes out as the wrong shape a lot.
    # this fixes it in a stupid yet effective way
    med_tifs = np.hstack([med_tifs, tifs])[:n_med_frames]
    print("{}: loading initial {} layers for median filter...".format(
        mp_id, med_tifs.size))
    med_cube = par_load_img(med_tifs)
    med = np.median(med_cube, axis=0)
    print("{}: median calculated".format(mp_id))

    # spoof Detector_Reducor stuff
    panel_shape = (n_frames,) + med.shape
    data = np.zeros(np.array(panel_shape) + (11, 0, 0), dtype=np.int16)
    spots = np.zeros([0, 4], dtype=float)
    coords = np.zeros([0, 3], dtype=np.int16)
    vals = np.zeros([0, 1], dtype=float)
    fids = np.zeros([1, 1], dtype=int)
    loaded = 0
    # we want to add an empty frame to the start so the spot finder sees
    # all the spots as "terminating" at the beginning. so, spoof a layer in
    # memory and set explored to -1 (cause we havent "explored" that empty
    # layer yet.)
    explored = -1
    in_memory = int(1)

    while loaded < n_frames:
        l_stop = np.min([loaded + 5, n_frames])
        i_start = in_memory
        i_stop = i_start + (l_stop - loaded)
        # load next chunk
        new = par_load_img(all_tifs[loaded:l_stop]) - med
        new[new < thresh] = 0
        data[i_start: i_stop] = new.copy()
        loaded = l_stop - 0
        in_memory = i_stop - 0
        if in_memory < 40 and l_stop < n_frames:
            continue
        if l_stop >= n_frames:
            # if this is the final run, grab 2 blank layers to ensure
            # cc3d properly terminates
            i_stop = np.min([i_stop + 2, data.shape[0]])
        # print("{}: - {} loaded, {} in memory. starting reduction".format(
        #     mp_id, loaded, i_stop))
        out = cc3d_nf_feature_finder(data[:i_stop], np.max(fids)+1)
        spots = np.vstack([spots, out[0]+[0, explored, 0, 0]])
        coords = np.vstack([coords, out[1]+[explored, 0, 0]])
        vals = np.vstack([vals, out[2]])
        fids = np.vstack([fids, out[3]])
        if explored == l_stop - out[4]:
            # shouldn't ever trigger
            print('wat?')
            continue
        deletable_layers = l_stop - out[4] - explored
        data = data[deletable_layers:]
        explored = l_stop - out[4]
        in_memory = out[4]
        # print("{}: -- {} layers purged ".format(mp_id, deletable_layers))
        print("{}: - {} checked, {} in reduction, {} purged".format(
            mp_id, loaded, i_stop, deletable_layers))

    # save to h5py
    first_part = chore['to'] + os.sep + chore_name
    h5_save_name_full = first_part + '.sparse'
    a = " ==================================== \n"
    b = "{}: +++ saving h5 to {}".format(mp_id, chore['to'])
    print(a + b + a)
    h5_save = h5py.File(h5_save_name_full, 'w')
    h5_save.create_dataset('spots', data=spots, compression='gzip')
    h5_save.create_dataset('coords', data=coords, compression='gzip')
    h5_save.create_dataset('vals', data=vals, compression='gzip')
    h5_save.create_dataset('fids', data=fids[1:], compression='gzip')
    for meta in ['epoch', 'z_height', 'nframes', 'from', 'omega', 'exposure']:
        h5_save.attrs[meta] = chore[meta]
    h5_save.attrs['xyo_dims'] = panel_shape
    h5_save.close()
    # also spoof framecaches
    spoof_frame_cache(first_part+".npz", coords, vals, panel_shape)
    return chore_name


def cv_load_wrapper(i, f):
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    return (im, i)


def par_load_img(files):
    executor = ProcessPoolExecutor(8)
    futures = {executor.submit(cv_load_wrapper, x[0], x[1]
                               ): x for x in enumerate(files)}
    first = True
    for future in as_completed(futures):
        out, i = future.result()
        if first:
            all_out = np.zeros((len(files),) + out.shape, dtype=out.dtype)
            first = False
        all_out[i] = out
        futures.pop(future)
    return all_out


def _spoof_frame_cache_single_processor(coords, vals):
    arrd = {}
    z = np.sort(coords[:, 0], kind='mergesort')
    for i in np.unique(z):
        mask = coords[:, 0] == i
        rc = coords[mask]
        d = (vals[mask]).flatten()
        arrd[f'{i}_row'] = rc[:, 1]
        arrd[f'{i}_col'] = rc[:, 2]
        arrd[f'{i}_data'] = d
    return arrd
