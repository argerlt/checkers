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
import cc3d
import sparse
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
# from multiprocessing import Pool
# from functools import reduce


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


def reduce_entire_ff(chore, chore_name, det_red, n_med_frames):
    mp_id = "p_"+str(os.getpid() % 41)
    a = " ==================================== \n"
    b = "    {}: starting {}\n".format(mp_id, chore_name)
    print(a + b + a)
    f_h5L = h5py.File(glob.glob(chore['from']+os.sep+"*ff1*")[0], 'r')
    f_h5R = h5py.File(glob.glob(chore['from']+os.sep+"*ff2*")[0], 'r')
    dat_h5L = f_h5L['imageseries/images']
    dat_h5R = f_h5R['imageseries/images']

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
        # k = [x for x in reducable][0]
        # a = cc3d_feature_finder(
        #       data[k][(explored[k]-min_explored):i_stop],1 , k)
        futures = {executor.submit(
            cc3d_feature_finder,
            data[k][(explored[k]-min_explored): i_stop],
            np.max(fids[k]+1),
            k):
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
        if min_explored >= l_stop-n_skips:
            print("XXXXX WARNING: I think I made a goof XXXXX")
        # print("back to loading from disk....")

    # save to h5py
    save_thresh = 250
    first_part = chore['to'] + os.sep + chore_name
    h5_save_name_full = first_part + '.sparse'
    h5_save_name_sml = first_part + 'T{}.sparser'.format(save_thresh)
    print("{}: +++ saving h5 to {}".format(mp_id, chore['to']))
    h5_save = h5py.File(h5_save_name_full, 'w')
    h5_save_sml = h5py.File(h5_save_name_sml, 'w')
    # make thresholded data
    masks = dict([(k, vals[k] > save_thresh) for k in keys])
    m_coords = dict([(k, coords[k][masks[k].flatten(), :]) for k in keys])
    m_vals = dict([(k, vals[k][masks[k]]) for k in keys])
    m_fids = dict([(k, fids[k][1:][masks[k]]) for k in keys])
    for k in keys:
        # all
        grp = h5_save.create_group(k)
        grp.create_dataset('spots', data=spots[k], compression='gzip')
        grp.create_dataset('coords', data=coords[k], compression='gzip')
        grp.create_dataset('vals', data=vals[k], compression='gzip')
        grp.create_dataset('fids', data=fids[k][1:], compression='gzip')
        # aggressively thresholded
        grp_sml = h5_save_sml.create_group(k)
        grp_sml.create_dataset('spots', data=spots[k], compression='gzip')
        grp_sml.create_dataset('coords', data=m_coords[k], compression='gzip')
        grp_sml.create_dataset('vals', data=m_vals[k], compression='gzip')
        grp_sml.create_dataset('fids', data=m_fids[k][1:], compression='gzip')
    for thing in ['load', 'epoch', 'z_height', 'nframes', 'from']:
        h5_save.attrs[thing] = chore[thing]
        h5_save_sml.attrs[thing] = chore[thing]
    h5_save.close()
    h5_save_sml.close()

    # also spoof framecaches
    executor = ThreadPoolExecutor(len(keys)*2)
    futures = {executor.submit(
        spoof_frame_cache,
        first_part+"-" + k + ".npz",
        coords[k],
        vals[k],
        subpanel_shape):
            k for k in keys}
    sparse_futures = {executor.submit(
        spoof_frame_cache,
        first_part+"-" + k + "T{}.npz".format(save_thresh),
        m_coords[k],
        m_vals[k],
        subpanel_shape):
            k for k in keys}
    fid_futures = {executor.submit(
        spoof_frame_cache_w_fid,
        first_part+"-fid-" + k + "T{}.npz".format(save_thresh),
        m_coords[k],
        m_vals[k],
        m_fids[k],
        subpanel_shape):
            k for k in keys}
    futures.update(sparse_futures)
    futures.update(fid_futures)
    for future in as_completed(futures):
        out = future.result()
        futures.pop(future)  # stop memory leak
        print("saved " + future.result())
    executor.shutdown()

    # cleanup to stop mem leak when multithreading
    del coords, vals, spots, fids, thresh, med, dataL, dataR
    del executor, data
    return
