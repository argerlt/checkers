#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:58:10 2024

@author: gerlt.1
"""

import numpy as np
import h5py
import glob
import os
import warnings
import cc3d
import sparse
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from hexrd.utils.hdf5 import unwrap_dict_to_h5

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

    def __init__(self, raw_dir, red_dir, name='spec.log', start_marker='#S'):
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
        self.make_task_list()

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

    def _parse_rams4_slew_ome(self, fid):
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
def cc3d_feature_finder(data_slice,
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
        f2v = np.where([feature_map[i]>0])
        if f2v[0].size <1:
            continue
        fm_flat = feature_map[i][f2v[1:]]
        fid, inv, count = np.unique(fm_flat, False, True, True)
        # toss 3x3 burned out pixels connecting spots accross omegas.
        fid[count < 10] = 0
        fid[fid > 0] = 1
        fid = fid.astype(np.int8)
        binarized[i][f2v[1:]] =fid[inv]
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
    binarized[np.isin(feature_map,finished_spots)] = 1
    feature_map = cc3d.connected_components(binarized)
    final_spot_ids = np.unique(feature_map[feature_map>0])
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
            continue # This spot was already fully observed. Ignore.
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
                     create_npz=True, spoof_thresholded=False,
                     save_threshold=250):
    mp_id = "p_"+str(os.getpid() % 41)
    a = " ==================================== \n"
    b = "    {}: starting {}\n".format(mp_id, chore_name)
    print(a + b + a)
    f_h5L = h5py.File(glob.glob(chore['from']+os.sep+"*ff1*")[0], 'r')
    f_h5R = h5py.File(glob.glob(chore['from']+os.sep+"*ff2*")[0], 'r')
    dat_h5L = f_h5L['imageseries/images']
    dat_h5R = f_h5R['imageseries/images']
    
    # because it's annoying to finish all the processing, only to fail during
    # saving, pre-flight the hdf5 save file.
    first_part = chore['to'] + os.sep + chore_name
    h5_save_name_full = first_part + '.sparse'
    h5_save = h5py.File(h5_save_name_full, 'w')
    for thing in ['load', 'epoch', 'z_height', 'nframes', 'from']:
        h5_save.attrs[thing] = chore[thing]
    settings_grp = h5_save.create_group('settings/initial_instr')
    unwrap_dict_to_h5(settings_grp, instr_dict)

    # begin loading frames for median filter
    n_frames, n_skips = det_red._parse_chore(chore)[1:]
    med_skip = np.floor(n_frames/n_med_frames).astype(int)
    medL = np.median(dat_h5L[4::med_skip], axis=0).astype(np.uint16)
    medR = np.median(dat_h5R[4::med_skip], axis=0).astype(np.uint16)
    med = dict()
    mean_med = dict()
    thresh = dict()
    for name, meds in ['ff1_', medL], ['ff2_', medR]:
        for key in det_red.rect.keys():
            [[a, b], [c, d]] = det_red.rect[key]
            nname = name+key
            med[nname] = meds[a:b, c:d]
            bg = np.mean(med[nname])
            mean_med[nname] = bg
            # assume background noise follows Poisson distribution, as per
            # Ralph's MTEX paper. set the threshold to the 99.99 %.
            # ie, 99.9% sure the data at each pixel is NOT purely background
            thresh[nname] = stats.poisson(bg).ppf(0.9999) - bg

    panel_shape = np.array(dat_h5L.shape) - (n_skips, 0, 0)
    subpanel_shape = panel_shape//[1, det_red.shape[0], det_red.shape[1]]
    # Kay, MutiThreading caused a memory leak, and multiprocessing doesn't
    # make sense for loading in this context, so loading is single threaded.
    data = dict([(k, np.zeros(
        subpanel_shape + (11, 0, 0), dtype=np.int16
        )) for k in med.keys()])
    # temp_check = dict([(k, np.zeros(
    #     subpanel_shape, dtype=np.uint8
    #     )) for k in med.keys()])
    spots = dict([(k, np.zeros([0, 4], dtype=float)) for k in med.keys()])
    coords = dict([(k, np.zeros([0, 3], dtype=np.int16)) for k in med.keys()])
    vals = dict([(k, np.zeros([0, 1], dtype=float)) for k in med.keys()])
    fids = dict([(k, np.zeros([1, 1], dtype=int)) for k in med.keys()])
    explored = dict([(k, int(-1)) for k in med.keys()])
    loaded = 0
    min_explored = int(-1)
    partially_explored = int(1)
    # get static unlinked list of dictionary keys
    keys = [x for x in data.keys()]
    # redone_counter = 0

    while loaded < n_frames:
        l_start = loaded + n_skips
        l_stop = np.min([l_start + 8, n_frames + n_skips])
        i_start = partially_explored
        i_stop = i_start + (l_stop - l_start)
        dataL = np.asanyarray(
            dat_h5L[l_start:l_stop], dtype=np.int16)[:, ::-1, :]
        dataR = np.asanyarray(
            dat_h5R[l_start:l_stop], dtype=np.int16)[:, :, ::-1]
        for dr_key in det_red.rect.keys():
            kl, kr = 'ff1_' + dr_key, 'ff2_' + dr_key
            [[a, b], [c, d]] = det_red.rect[dr_key]
            dl = dataL[:, a:b, c:d] - med[kl]
            dr = dataR[:, a:b, c:d] - med[kr]
            dl[dl < thresh[kl]] = 0
            dr[dr < thresh[kr]] = 0
            data[kl][i_start:i_stop] = dl
            data[kr][i_start:i_stop] = dr
            # temp_check[kl][l_start - n_skips:l_stop - n_skips] = dl > 0
            # temp_check[kr][l_start - n_skips:l_stop - n_skips] = dr > 0
        loaded = l_stop - n_skips
        partially_explored = i_stop
        # print("{}: loading {} to {}, with {} in memory".format(
        #     mp_id, l_start, l_stop, i_stop))
        if partially_explored < 40 and l_stop < n_frames:
            continue  # break out if there isn't enough to bother reducing
        # print(" -- enough collected to start reduction")
        # print(" -- reductions will start at {} in l.".format(
        #     np.array([x for x in explored.values()])))
        # print(" -- max l = {}.".format(l_stop-n_skips))
        # print(" -- first {} slices already deleted.".format(min_explored))
        # print(" -- next {} previously looked at.".format(redone_counter))
        if l_stop < n_frames:
            n = l_stop - n_skips
            reducable = [x for x in keys if (n - explored[x]) > 20]
            if len(reducable) < 1:
                print("wat?")
                continue
        else:
            # get total layer count in any panel: can't go over this.
            i_total = np.max([data[k].shape[0] for k in keys])
            # set i_stop to catch all the data plus two zero layers, or
            # i_total, whichever is less
            i_stop = np.min([i_stop + 2, i_total])  # stopgap for partials
            # this time, since it's the last iteration, reduce every panel
            reducable = [x for x in keys]
        # executor = ThreadPoolExecutor(8)
        executor = ProcessPoolExecutor(8)
        # print("{}: -- running reduction ... ".format(mp_id))
        print("{}: - {} to {} loaded, {} in memory. starting reduction".format(
            mp_id, l_start, l_stop, i_stop))
        k = [x for x in reducable][0]
        a = cc3d_feature_finder(
              data[k][(explored[k]-min_explored):i_stop],1 , k)
        futures = {executor.submit(
            cc3d_feature_finder,
            data[k][(explored[k]-min_explored): i_stop],
            np.max(fids[k]+1),
            k,
            partially_explored-explored[k]):
                k for k in reducable}
        for future in as_completed(futures):
            out = future.result()
            futures.pop(future)  # stop memory leak
            k = out[-1]
            spots[k] = np.vstack([spots[k], out[0]+[0, explored[k], 0, 0]])
            coords[k] = np.vstack([coords[k], out[1]+[explored[k], 0, 0]])
            vals[k] = np.vstack([vals[k], out[2]])
            fids[k] = np.vstack([fids[k], out[3]])
            explored[k] = l_stop - out[4]
        executor.shutdown()
        # print(" -- done ")
        new_min = np.min([x for x in explored.values()])
        if new_min == min_explored:
            continue
        deleted = new_min - min_explored
        for k in keys:
            data[k] = data[k][deleted:]
        print("{}: -- {} layers done and purged ".format(mp_id, deleted))
        # redone_counter = partially_explored - deleted
        partially_explored = i_stop - deleted
        min_explored = new_min*1
        if min_explored > l_stop-n_skips:
            if loaded < n_frames:
                print("XXXXX WARNING: I think I made a goof XXXXX")
            else:
                print("{}: -- {} reduction completed".format(mp_id, deleted))

    # save to h5py
    print("{}: +++ saving h5 to {}".format(mp_id, chore['to']))
    for k in keys:
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
        masks = dict([(k, vals[k] > save_threshold) for k in keys])
        m_coords = dict([(k, coords[k][masks[k].flatten(), :]) for k in keys])
        m_vals = dict([(k, vals[k][masks[k]]) for k in keys])
        m_fids = dict([(k, fids[k][1:][masks[k]]) for k in keys])
        for k in keys:
            gs = h5_save_sml.create_group('data/' + k)
            gs.create_dataset('spots', data=spots[k], compression='gzip')
            gs.create_dataset('coords', data=m_coords[k], compression='gzip')
            gs.create_dataset('vals', data=m_vals[k], compression='gzip')
            gs.create_dataset('fids', data=m_fids[k][1:], compression='gzip')
        for thing in ['load', 'epoch', 'z_height', 'nframes', 'from']:
            h5_save_sml.attrs[thing] = chore[thing]
        settings_grp = h5_save_sml.create_group('settings/initial_instr')
        unwrap_dict_to_h5(settings_grp, instr_dict)
        h5_save_sml.close()

    # also spoof framecaches if requested
    if create_npz:
        executor = ThreadPoolExecutor(len(keys)*2)
        futures = {executor.submit(
            spoof_frame_cache,
            first_part+"-" + k + ".npz",
            coords[k],
            vals[k],
            subpanel_shape):
                k for k in keys}
        fid_futures = {executor.submit(
            spoof_frame_cache_w_fid,
            first_part+"-fid-" + k + ".npz",
            coords[k],
            vals[k],
            fids[k][1:],
            subpanel_shape):
                k for k in keys}
        futures.update(fid_futures)
        if spoof_thresholded:
            sparse_futures = {executor.submit(
                spoof_frame_cache,
                first_part+"-" + k + "T{}.npz".format(save_threshold),
                m_coords[k],
                m_vals[k],
                subpanel_shape):
                    k for k in keys}
            futures.update(sparse_futures)
            sparse_fid_futures = {executor.submit(
                spoof_frame_cache_w_fid,
                first_part+"-fid-" + k + "T{}.npz".format(save_threshold),
                m_coords[k],
                m_vals[k],
                m_fids[k],
                subpanel_shape):
                    k for k in keys}
            futures.update(sparse_fid_futures)
        for future in as_completed(futures):
            out = future.result()
            futures.pop(future)  # stop memory leak
            print("saved " + future.result())
        executor.shutdown()

    # cleanup to stop mem leak when multithreading
    del coords, vals, spots, fids, thresh, med, dataL, dataR
    del executor, data
    return
