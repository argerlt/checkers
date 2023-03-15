#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:05:28 2023

@author: Austin Gerlt, Simon Mason
Code for preprocessing nf data using both parallelized loading and nl-means
"""
# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import json
import re
import warnings
import time

# image processing stuff
from skimage import io, filters
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy import ndimage

# multithreading libraries
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


# =============================================================================
# %% set some initial file paths and other per-run editable variables
# =============================================================================
# nf_raw_folder = "/fs/scratch/PAS2405/CHESS/raw/2023_1/*me3*nf*"
nf_raw_folder = 'C:\\Users\\agerlt\\workspace\\globus\\2023_1_nf'
#reduced_dir = "/fs/scratch/PAS2405/CHESS_2023_1/reduced"
reduced_dir = "C:\\Users\\agerlt\\workspace\\SL\\SL_nf"
output_file_name = "lets_test_this"
stack_ID = 1
binarization_threshold = 20
# set test_only = True if you do NOT want to save out any data, and instead
# want to plot the results of some cleaning algorithms on your sample.
test_only = False
dynamic_median_filter = True
save_non_binarized = False
# =============================================================================
# %% Some general use metadata skimming functions
# =============================================================================
# NOTE: I'm positive there is a more standard way CHESS users do what these
# functions do, but these work for me. Feel free to replace them with your own
# bespoke skimmers


def skim_metadata(raw_folder, cycle="2023_1", output_dict=False):
    """
    skims all the .josn and .par files in a folder, and returns a concacted
    pandas DataFrame object with duplicates removed. If Dataframe=False, will
    return the same thing but as a dictionary of dictionaries.

    NOTE: uses Pandas Dataframes because some data is int, some float, some
    string. Pandas auto-parses dtypes per-column, and also has
    dictionary-like indexing.
    """
    # grab all the json files, assert they both exist and have par pairs
    f_jsons = glob.glob(raw_folder + os.sep + "*json")
    assert len(f_jsons) > 0, "No .jsons found in {}".format(nf_raw_folder)
    f_par = [x[:-4] + "par" for x in f_jsons]
    assert np.all([os.path.isfile(x) for x in f_par]), "missing .par files"
    # read in headers from jsons
    headers = [json.load(open(j, "r")).values() for j in f_jsons]
    # read in headers from each json and data from each par as Dataframes
    df_list = [
        pd.read_csv(p, names=h, delim_whitespace=True, comment="#")
        for h, p in zip(headers, f_par)
    ]
    # concact into a single dataframe and delete duplicate columns
    meta_df_dups = pd.concat(df_list, axis=1)
    meta_df = meta_df_dups.loc[:, ~meta_df_dups.columns.duplicated()].copy()
    if output_dict:
        # convert to dict of dicts if requested
        return dict(zip(meta_df.keys(), [x[1].to_list() for x in meta_df.iteritems()]))
    # else, just return
    return meta_df


def ome_from_df(meta_df):
    """ takes in a dataframe generated from the metadata using "skim_metadata",
    and returns a numpy array of the omega data (ie, what frames represent
    which omega angle in the results)"""
    start = meta_df["ome_start_req"].to_numpy()
    stop = meta_df["ome_end_req"].to_numpy()
    steps = meta_df["nframes_real"].to_numpy()
    scan = meta_df["SCAN_N"].to_numpy()
    lines = [np.linspace(a, b - (b - a) / c, c)
             for a, b, c in zip(start, stop, steps)]
    omes = np.hstack([x for y, x in sorted(zip(scan, lines))])
    # sanity checks
    if len(np.unique([len(x) for x in lines])) != 1:
        warnings.warn("omega data chunks are of uneven size", UserWarning)
    if len(np.unique(np.around(omes[1:] - omes[:-1], 5))) != 1:
        print(np.unique(np.around(omes[1:] - omes[:-1], 5)))
        warnings.warn("omega data has uneven deltas", UserWarning)
    return omes


def image_locs_from_df(meta_df, raw_folder):
    """ takes in a dataframe generated from the metadata using "skim_metadata"
    plus the near_field folder locations, and returns a list of image locations
    """
    scan = meta_df["SCAN_N"].to_numpy()
    first = meta["first_image_scan"].to_numpy()
    last = meta["last_image_scan"].to_numpy()
    # note to future readers: 'goodstart' is the first usable image,
    # and sometimes there is junk images at start AND end, so
    # be sure to remove off both sides. this "first/last" trick works for
    # 2022_3 and 2023_1, might not work in future though.

    files = []
    for i in range(len(scan)):
        all_files = glob.glob(raw_folder + os.sep + str(scan[i]) + "/nf/*.tif")
        all_names = [x.split(os.sep)[-1] for x in all_files]
        all_ids_list = [int(re.findall("([0-9]+)", x)[0]) for x in all_names]
        all_ids = np.array(all_ids_list)
        good_ids = (all_ids >= first[i]) * (all_ids <= last[i])
        files.append([x for x, y in sorted(zip(all_files, good_ids)) if y])
    # flatten the list of lists
    files = [item for sub in files for item in sub]
    # sanity check
    s_id = np.array(
        [int(re.findall("([0-9]+)", x.split(os.sep)[-3])[0]) for x in files]
    )
    f_id = np.array(
        [int(re.findall("([0-9]+)", x.split(os.sep)[-1])[0]) for x in files]
    )
    assert np.all(s_id[1:] - s_id[:-1] >= 0), "folders are out of order"
    assert np.all(f_id[1:] - f_id[:-1] >= 0), "files are out of order"

    return files


# =============================================================================
# %% Functions for doing threaded processes
# =============================================================================


def img_in(slice_id, filename):
    """ This just wraps io.imread, but allows the slice_id to be an unused
    input. this makes reordering the threaded data easier"""
    return io.imread(filename)


def dynamic_median(z, median_distance):
    """subtracts a local median darkfield from a slice using plus-or-minus a
    number of frames in the omega equal to "median_distance".
    Note, this is NOT a per-image process, it's taking a line from each frame,
    and comparing adjacent frames to each other to allow for multithreading"""
    plate = img_cube[:, :, z]
    local_dark = ndimage.median_filter(plate, size=[median_distance, 1])
    new_plate = plate-local_dark
    new_plate[local_dark >= plate] = 0
    return new_plate


def static_median(z, nth):
    """subtracts a median darkfield background, where the darkfield is
    calculated using every "nth" frame.
    Note, this is NOT a per-image process, it's taking a line from each frame,
    and comparing adjacent frames to each other to allow for multithreading"""
    plate = img_cube[:, :, z]
    darkfield = np.median(plate[::nth, :], axis=0)
    # fix this. it makex negatives real big bc uint overflow.
    new_plate = plate-darkfield
    new_plate[darkfield >= plate] = 0
    return new_plate


def per_slice_nl_means(slice_id, thresh=4, size=5, dist=5):
    """
    Simon Mason figured out this function, comments are adopedet from him.
    Takes in a slice id, then loads and processes it in a thread-friendly
    way
    """
    img = img_cube[slice_id, :, :]
    # estimage the per-slice sigma
    s_est = np.mean(estimate_sigma(img))
    # run nl_means
    # size is kernel size of patch for denoising
    # dist is maximal distance (in pixels from center) for similar patches
    #  sigma should be scaled smaller than sigma to avoid oversmoothing
    img = denoise_nl_means(
        img, sigma=s_est, h=0.8 * s_est, patch_size=size, patch_distance=dist
    )
    img_binary = img > thresh
    # Final dilation. IDK why this is here, but it's in every HEXRD preprocess
    img_binary = binary_dilation(img_binary, iterations=1)
    return slice_id, img, img_binary


def per_slice_hexrd_gaussian_cleanup(slice_id, thresh=4, sigma=4, size=5):
    """Verbaitm copy of the gaussian cleanup used in hexrd.grainmap.nfutil,
    just in a function that is easier to multithread"""
    img = img_cube[slice_id, :, :]
    img = filters.gaussian(img, sigma=sigma)
    img = ndimage.morphology.grey_closing(img, size=(size, size))
    img_binary = img > thresh
    # Final dilation. IDK why this is here, but it's in every HEXRD preprocess
    img_binary = binary_dilation(img_binary, iterations=1)
    return slice_id, img, img_binary


def per_slice_hexrd_erosion_cleanup(slice_id, thresh=4, erosions=4, dilations=5):
    """Verbaitm copy of the erosion cleanup used in hexrd.grainmap.nfutil,
    just in a function that is easier to multithread"""
    img = img_cube[slice_id, :, :]
    img_binary = img > thresh
    img_binary = binary_erosion(img_binary, iterations=erosions)
    img_binary = binary_dilation(img_binary, iterations=dilations)
    # Final dilation. IDK why this is here, but it's in every HEXRD preprocess
    img_binary = binary_dilation(img_binary, iterations=1)
    return slice_id, img, img_binary


def per_slice_austin(slice_id, thresh=2, nl_agg=1.2, er=5, di=4):
    """My best try, hybrid model using a median filter. works fast, seems to
    handle pixel error well. Alternate hybrid using nl-means is shown at the
    end of this document, but it scales geometrically with image size, and is
    thus pretty slow."""
    img = img_cube[slice_id, :, :]
    med = ndimage.median_filter(img, size=[5, 5])
    closed = ndimage.morphology.grey_closing(med, size=(5, 5))
    binarized = closed > thresh
    final = binary_dilation(binary_erosion(
        binarized, iterations=er), iterations=di)
    return slice_id, closed, final


def dummy(x):
    """Dumb but functional way to clear out the ThreadExecutor"""
    return "dummy"


# ==============================================================================
# %% Collect Metadata and file locations
# ==============================================================================
# comb the nf folder for metadata files and compile them
all_meta = skim_metadata(nf_raw_folder)
# find the unique z_height requests, parse out just the a specific layer
unique_zheights = np.sort(all_meta["rams4z"].unique())
target_zheight = unique_zheights[stack_ID]
meta = all_meta[all_meta["rams4z"] == target_zheight]
# get the array of per-frame omega values and file locations
omegas = ome_from_df(meta)
f_imgs = image_locs_from_df(meta, nf_raw_folder)
# create an Image Collection, which is basically a hexrd.ImageSeries but better
# NOTE: if you are a hexrd traditionalist and prefer imageseries, you'll
# want to change this line accordingly
img_collection = io.imread_collection(f_imgs)

# Sanity check
assert len(omegas) == len(
    img_collection
), """
    mismatch between expected and actual filecount"""


# ==============================================================================
# %% load and dark-field subtract all the data
# ==============================================================================


global img_cube
# img_collection = img_collection[:100] # this is the line to change if you want to do a truncated test
# load the very first image just to get the dimensions
y, z = img_collection[0].shape
img_cube = np.zeros([len(img_collection), y, z], dtype=np.uint16)
bin_cube = np.zeros([len(img_collection), y, z], dtype=np.bool8)

# load everything
print("loading images...")
with ThreadPoolExecutor(50) as executor:
    tic = time.time()
    id_and_file = [x for x in enumerate(img_collection.files)]
    futures = {executor.submit(img_in, x[0], x[1]): x for x in id_and_file}
    i = 0
    for future in as_completed(futures):
        data = future.result()
        inputs = futures.pop(future)  # this also stops memory leaks
        slice_id = inputs[0]
        img_cube[slice_id] = data
        i += 1
        print("{} of {} images loaded".format(i, img_cube.shape[0]))
        del inputs, slice_id, data
    print("Everything is loaded in")
    tocA = time.time() - tic
    print(tocA)

# perform median subtraction
    zz = np.arange(img_cube.shape[2])
    if dynamic_median_filter:
        # Futures code for dynamic median filter
        futures = {executor.submit(dynamic_median, z, 15): z for z in zz}
        print("Starting dynamic median dark field subtraction...")
    else:
        # Futures code for static median filter
        futures = {executor.submit(static_median, z, 50): z for z in zz}
        print("Starting static median dark field subtraction...")
    i = 0
    for future in as_completed(futures):
        new_plate = future.result()
        z = futures.pop(future)  # this also stops memory leaks
        img_cube[:, :, z] = new_plate
        i += 1
        if i % 25 == 0 or i == zz[-1]:
            print("{} of {} y-slices filtered".format(i, img_cube.shape[2]))
        del z, new_plate

    print("Dark field subtraction is done")
    tocB = time.time() - tic - tocA
    print(tocB)
    executor.shutdown()

# save out test_cube for easier plotting later
if test_only:
    np.save("test_block", img_cube[20:40, :1200, :])
# ==============================================================================
# %% if requested, clean and save out the image cube
# ==============================================================================
# cleaned_cube = np.zeros(img_cube.shape,dtype=np.float32)

if not test_only:
    executor = ThreadPoolExecutor(40)
    xx = np.arange(img_cube.shape[0])
    # futures = {executor.submit(per_slice_nl_means,x, binarization_threshold, 5, 5): x for x in xx}
    # futures = {executor.submit(per_slice_hexrd_gaussian_cleanup, x, binarization_threshold, 4, 5): x for x in xx}
    # futures = {executor.submit(per_slice_hexrd_erosion_cleanup, x, binarization_threshold, 4, 5): x for x in xx}
    # thread_friendly copy of Austin's cleanup
    futures = {executor.submit(per_slice_austin, x): x for x in xx}
    print("starting denoising and binarization...")
    i = 0
    for future in as_completed(futures):
        x, cleaned_slice, binarized_slice = future.result()
#        cleaned_cube[x, :, :] = cleaned_slice
        bin_cube[x, :, :] = binarized_slice
        inputs = futures.pop(future)  # this stops memory leaks
        i += 1
        if i % 25 == 0 or i == xx[-1]:
            print("{} of {} layers cleaned".format(i, img_cube.shape[0]))
        del inputs, x, cleaned_slice, binarized_slice  # this is just a cleanup
    tocC = time.time() - tic - tocA - tocB
    print(tocC)
    executor.shutdown()

    # Save the results
    np.save(reduced_dir + os.sep + output_file_name + "_binarized", bin_cube)
    if save_non_binarized:
        np.save(reduced_dir + os.sep + output_file_name, img_cube)


# ==============================================================================
# %% Bonus: Testing effects of different cleaning algorithms
# ==============================================================================
# The rest of this code doesn't need to run, it's only purpose is to test out
# how different cleaning algorithms seem to do.

if test_only:
    # backup line to load in code if you make a boo-boo
    test_block = np.load('test_block.npy')
    img = test_block[8, 200:800, 200:800]
    s_est = np.mean(estimate_sigma(img))
    nl_means_a = denoise_nl_means(img, 5, 5, 0.4*s_est)
    nl_means_b = denoise_nl_means(img, 5, 11, 1.2*s_est)
    hexrd_gaussian = ndimage.morphology.grey_closing(
        filters.gaussian(img, 5), (5, 5)
        )
    hexrd_erosion = binary_dilation(
        binary_erosion(img, iterations=5), iterations=5
        )
    filtered = filters.gaussian(denoise_nl_means(img, 5, 11, 1.2*s_est), 1.5)
    binarized = filtered > 2
    nl_err_dil = binary_dilation(
        binary_erosion(binarized, iterations=5), iterations=4
        )

    plt.close('all')

    plt.figure()
    plt.title('original')
    plt.imshow(img, vmax=10)

    plt.figure()
    plt.title('nl_means_a')
    plt.imshow(nl_means_a, vmax=10)

    plt.figure()
    plt.title('nl_means_b')
    plt.imshow(nl_means_b, vmax=10)

    plt.figure()
    plt.title('hexrd_gaussian')
    plt.imshow(hexrd_gaussian)

    plt.figure()
    plt.title('hexrd_erosion')
    plt.imshow(hexrd_erosion)

    plt.figure()
    plt.title('nl_plus_errosion_dilation')
    plt.imshow(nl_err_dil)
