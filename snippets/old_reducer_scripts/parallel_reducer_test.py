#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:50:19 2024

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
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from functools import reduce


global data
global med
global thresh

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
                 panel_names = ['ff1','ff2']
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


def single_layer_load_LR(h5_locL, h5_locR, i, dr, skips=4):
    out = dict()
    layerL = np.asanyarray(h5_locL[i + skips], dtype=np.uint16)[::-1, :]
    for key in dr.rect.keys():
        [[a, b], [c, d]] = dr.rect[key]
        out['ff1_' + key] = layerL[a:b, c:d]
    del layerL
    layerR = np.asanyarray(h5_locR[i + skips], dtype=np.uint16)[:, ::-1]
    for key in dr.rect.keys():
        [[a, b], [c, d]] = dr.rect[key]
        out['ff2_' + key] = layerR[a:b, c:d]
    del layerR
    return out, i


def single_layer_load_LR_global(h5_locL, h5_locR, i, iic, dr, skips=4):
    layerL = np.asanyarray(h5_locL[i + skips], dtype=np.uint16)[::-1, :]
    for key in dr.rect.keys():
        [[a, b], [c, d]] = dr.rect[key]
        k = 'ff1_' + key
        data[k][iic] = np.max([layerL[a:b, c:d], med[k]], axis=0) - med[k]
        data[k][iic][data[k][i+1]< thresh[k]] =0
    del layerL
    layerR = np.asanyarray(h5_locR[i + skips], dtype=np.uint16)[:, ::-1]
    for key in dr.rect.keys():
        [[a, b], [c, d]] = dr.rect[key]
        k = 'ff2_' + key
        data[k][iic] = np.max([layerR[a:b, c:d], med[k]], axis=0) - med[k]
        data[k][iic][data[k][i+1]< thresh[k]] =0
    return i, iic


def cc3d_feature_finder(data_slice,
                        smallest_free_fid,
                        k,
                        min_spot_size=50,
                        min_total_spot_intensity=10000,
                        ):
    # get binarized yes/no, use it to assign spot IDs.
    binarized = (data_slice > 0).astype(np.int8)
    feature_map = cc3d.connected_components(binarized)
    del binarized
    # get the id's of the spots that go into the next unlooked at area
    unfinished_spots, ufc = np.unique(feature_map[-1], False, False, True)
    # get the id's of already documented spots
    old_spots = np.unique(feature_map[0])
    # toss out ones with fewer than 5 pixels of data (almost always noise)
    unfinished_spots = unfinished_spots[ufc > 4]
    # clump the feature maps by id.
    fids, idxs, inv, counts = np.unique(feature_map[:-1], True, True, True)

    # find first layer with an unfinished grain in it
    unfinished_fids = idxs[np.isin(fids, unfinished_spots[1:])]
    # rarely, a stringer will go through every layer. ignore them.
    pix_per_layer = feature_map[0].size
    unfinished_fids = unfinished_fids[unfinished_fids > pix_per_layer]
    # floor divide to get highest layer with no incomplete spots
    if unfinished_fids.size == 0:
        redo_count = 0
    else:
        redo_count = data_slice.shape[0] - np.min(unfinished_fids // pix_per_layer)

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
    del cleaned_fm, cleaned_fids, finished_spots, big_spots, old_spots
    del data_slice, ufc, unfinished_spots, unfinished_fids
    if spot_id < 1:  # triggers if no spots were found
        del sparse_fm
        d1 = np.zeros([0, 4], dtype=float)
        d2 = np.zeros([0, 3], dtype=np.int16)
        d3 = np.zeros([0, 1], dtype=float)
        d4 = np.zeros([0, 1], dtype=int)
        return d1, d2, d3, d4, redo_count, k
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
    vals = np.atleast_2d(sparse_fm.data[mask]).T
    fids = np.atleast_2d(final_fids[mask]).T
    del sparse_fm
    return spots, crds, vals, fids, redo_count,k


def spoof_frame_cache(name,coords,vals):
    arrd = {}
    for i in np.unique(coords[:,0]):
        rc = coords[coords[:,0] == i]
        d = vals[coords[:,0] == 534]
        arrd[f'{i}_row'] = rc[:,0]
        arrd[f'{i}_col'] = rc[:,1]
        arrd[f'{i}_data'] = d
    arrd['shape'] = coords.shape
    arrd['nframes'] = coords.shape[0]
    arrd['dtype'] = str(d.dtype).encode()
    np.savez_compressed(name,**arrd)
    return
# %%


parent_raws = "/local/scratch2/CHESS_data/raw/2022-3/id1a3/ko-3371-b/"
top_red = "/local/scratch2/CHESS_data/reduced/2022_me36_ff_sparse"
lr = LogReader(parent_raws+"me3-6_fatigue", top_red)
t1 = lr.task_list
t2 = LogReader(parent_raws+"me3-6-unloaded-ff-1", top_red).task_list
t3 = LogReader(parent_raws+"me3-6-fatigue-unloaded-ff-1", top_red).task_list
tasks = reduce(lambda a,b:{**a, **b}, [t1, t2, t3])

# few tests were tiny scans to see if the part was still there. ignore these.
keys = [x for x in tasks.keys()]
for key in keys:
    if tasks[key]['nframes']<1440:
        tasks.pop(key)
        print(key)

# I want the loads at each step too, so we grab that from the pso log.s
# grab the epoch and z height as well from the meta scalars.txt
keys = [x for x in tasks.keys()]
for key in keys:
    fname = glob.glob(tasks[key]['from'][:-2]+"/pso/*log.txt")[0]
    with open(fname,'r') as f:
        txt = [f.readline()[:-2] for x in range(10)]
        f.close()
    load_txt = [x.split(",")[0] for x in ",".join(txt).split("(N)=")[1:]]
    load = np.array(load_txt,dtype=float).mean()
    tasks[key]['load'] = load
    arr = np.loadtxt(tasks[key]['from'][:-2]+"/meta/1/scalars.txt")
    tasks[key]['epoch'] = np.mean(arr[:,2])
    tasks[key]['z_height'] =np.mean(arr[:,4])
    
# finally, sort keys into chronological order
chore_names = np.array(keys)[np.argsort([tasks[x]['epoch'] for x in keys])]


# %%
# test run


# %%

ex_load = ThreadPoolExecutor(8)
ex_process = ThreadPoolExecutor(8)

for chore_name in chore_names:
    chore = tasks[chore_name]
    print(chore_name)
    dr = Detector_Reducer()
    ff = [glob.glob(chore['from']+os.sep+"*{}*".format(x))[0] for x in dr.p_names]
    f_h5L = h5py.File(ff[0], 'r')
    f_h5R = h5py.File(ff[1], 'r')
    dat_h5L = f_h5L['imageseries/images']
    dat_h5R = f_h5R['imageseries/images']
    
    n_med_frames = 21
    n_frames, n_skips = dr._parse_chore(chore)[1:]
    med_skip = np.floor(n_frames/n_med_frames).astype(int)
    medL = np.median(dat_h5L[4::med_skip], axis=0).astype(np.uint16)
    medR = np.median(dat_h5R[4::med_skip], axis=0).astype(np.uint16)
    med = dict()
    mean_med = dict()
    thresh = dict()
    for name, meds in ['ff1_', medL], ['ff2_', medR]:
        for key in dr.rect.keys():
            [[a, b], [c, d]] = dr.rect[key]
            nname = name+key
            med[nname] = meds[a:b, c:d]
            bg = np.mean(med[nname])
            mean_med[nname] = bg
            # assume background noise follows Poisson distribution, as per
            # Ralph's MTEX paper. set the threshold to the 99.9 %.
            # ie, 99.9% sure the data at each pixel we see is NOT purely background
            thresh[nname] = stats.poisson(bg).ppf(0.999)-bg
    
    panel_shape = np.array(dat_h5L.shape) - (n_skips,0,0)
    subpanel_shape = panel_shape//[1,dr.shape[0],dr.shape[1]]
    # Kay, MutiThreading caused a memory leak, and multiprocessing doesn't
    # make sense for loading in this context, so loading is single threaded.

    
    loaded = np.arange(n_frames+1)
    loaded[0] = 1E6
    # THEN multithread the loading and subtracting
    # NOTE: THiS IS MOSTLY GOING TO BE SINGLE THREADED ANYWAY. the process is
    # NOT CPU bound, it's IO bound. the multithreading just does the median,
    # threshold, and featureID while waiting on future layers to load.
    # To get a meaningful speedup beyond this would require changing hardware.
    data = dict([(k, np.zeros(subpanel_shape + (11, 0, 0))) for k in med.keys()])
    last_incomplete_layer = 1
    spots = dict([(k, np.zeros([0, 4], dtype=float)) for k in med.keys()])
    coords = dict([(k, np.zeros([0, 3], dtype=np.int16)) for k in med.keys()])
    vals = dict([(k, np.zeros([0, 1], dtype=float)) for k in med.keys()])
    fids = dict([(k, np.zeros([1, 1], dtype=int)) for k in med.keys()])
    explored_layers = dict([(k, 0) for k in med.keys()])
    # get static unlinked list of dictionary keys
    keys = [x for x in data.keys()]


    futures = {ex_load.submit(single_layer_load_LR, dat_h5L, dat_h5R, i, dr):
                i for i in np.arange(n_frames)+n_skips
                }
    # futures = {ex_load.submit(single_layer_load_LR_global,
    #                           dat_h5L,
    #                           dat_h5R,
    #                           i+n_skips,
    #                           i + 1,
    #                           dr):
    #            i for i in np.arange(n_frames)
    #            }
    for future in as_completed(futures):
#        i, i_in_data_cube = future.result()
        layers, i = future.result()
        i_in_data_cube = i+1-n_skips
        for k in keys:
            layers[k] = np.max([layers[k], med[k]], axis=0) - med[k]
            layers[k][layers[k] < thresh[k]] = 0  # cut off poisson noise
            data[k][i_in_data_cube] = layers[k]
        loaded[i_in_data_cube] = 1E6
        print(i)
        z = futures.pop(future)  # stops memory leak
        # then, every 30 layers, run the feature finder
        consec_loaded = np.min(loaded)
        min_explored_layers = np.min([x for x in explored_layers.values()])
        if consec_loaded-min_explored_layers < 40 and consec_loaded < n_frames:
            continue
        slices = dict([(k, data[k][explored_layers[k]:consec_loaded]) for k in keys])
        if consec_loaded == n_frames:
            slices = dict([(k, data[k][explored_layers[k]:]) for k in keys])
        print(" -- {} loaded, looking from {}".format(
            consec_loaded, np.min([x for x in explored_layers.values()])))
        print(consec_loaded)
        print([slices[k].shape for k in keys])
    #     f2s = {ex_process.submit(cc3d_feature_finder,
    #                              slices[k],
    #                              np.max(fids[k]+1),
    #                              k):
    #            k for k in keys if slices[k].shape[0] > 10}
    #     for f2 in as_completed(f2s):
    #         out = f2.result()
    #         f2s.pop(f2)  # stop memory leak
    #         k = out[-1]
    #         spots[k] = np.vstack([spots[k], out[0]+[0, min_explored_layers,0,0]])
    #         coords[k] = np.vstack([coords[k], out[1]+[min_explored_layers,0,0]])
    #         vals[k] = np.vstack([vals[k], out[2]])
    #         fids[k] = np.vstack([fids[k], out[3]])
    #         explored_layers[k] = consec_loaded - out[4]
    
    #     #     new_sparse = cc3d_feature_finder(data[consec_ided:consec_loaded])
    #     #   spot_ids += smallest_free_fid
    #     #    spot_centers += [0, smallest_free_fid, 0, 0]
    # # Note to future me: this LOOKS like it's leaking, but its not.
    # # https://github.com/python/cpython/issues/98467
    
    # h5_save = h5py.File(chore['to']+os.sep+chore_name+'.sparse','w')
    # for k in keys:
    #     grp = h5_save.create_group(k)
    #     grp.create_dataset('spots', data=spots[k])
    #     grp.create_dataset('coords', data=coords[k])
    #     grp.create_dataset('vals', data=vals[k])
    #     grp.create_dataset('fids', data=fids[k])
    #     # also spoof framecache while here
    #     npz_name = chore['to']+os.sep+chore_name+"-"+k+".npz"
    #     spoof_frame_cache(npz_name, coords[k], vals[k])
    # for thing in ['load', 'epoch', 'z_height', 'nframes','from']:
    #     h5_save.attrs[thing] = chore[thing]
    # h5_save.close()



ex_load.shutdown()
ex_process.shutdown()






