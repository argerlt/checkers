#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:08:02 2024

@author: gerlt.1
"""

import numpy as np
import glob
import time
from pr_functions import LogReader
from pr_functions import Detector_Reducer
from pr_functions import reduce_entire_nf
import socket
import re
import sys
import os
import yaml

# NOTE TO ANYONE IN THE FUTURE (OR MAYBE FUTURE AUSTIN):
# This next if/elif is idiotic and overbuild, but i wanted to be able to
# run EXACTLY the same file on OSC and my laptop, to avoid version errors.
# this was the fastest way to do it. Better method for others though is to
# just delete the whole block of text

overwrite = True

##################################
## SET SOURCES AND DESTINATIONS ##
##################################

# add hard linked locations here
# ================
# loc = ['path/to/raw','path/to/reduced','path/to/meta']
# ================

# If no loc already, Figure out where I am, and set directories accordingly
computer_name = socket.gethostname()
SLURM = None # flag for switching forom serial to HPC embarassingly parallel
if re.match("^mse-dnc.*.coeit.osu.edu$", computer_name):
    print("running on KyloRam")
    locs = ["/local/scratch2/CHESS_data/raw/2022-3/id1a3/ko-3371-b/",
            "/local/scratch/2022_ME36_nf_ff_sparsified/ff/",
            "/local/scratch/2022_ME36_nf_ff_sparsified/metadata/"]
elif re.match("^o.*ten.osc.edu$", computer_name):
    print("running on OSC (Owens)")
    SLURM = "Owens"
    group_folder = "/fs/ess/PAS2405/CHESS_data/"
    locs = [group_folder + "raw/2022-3/id1a3/ko-3371-b/",
            group_folder + "2022_me36_ff_sparse/results/",
            group_folder + "metadata/"]
elif re.match("^p.*ten.osc.edu$", computer_name):
    print("running on OSC (Pitzer)")
    SLURM = "Pitzer"
    group_folder = "/fs/ess/PAS2405/CHESS_data/"
    locs = [group_folder + "raw/2022-3/id1a3/ko-3371-b/",
            group_folder + "2022_me36_ff_sparse/results/",
            group_folder + "metadata/"]
elif re.match("HAL", computer_name):
    print("running on Austin's Laptop")
    top_dir = "C:\\Users\\agerlt\\Data\\CHESS_data\\"
    locs = [
     top_dir + "2022_me36_ff_sparse\\raw\\",
     top_dir + "2022_ME36_nf_ff_sparsified\\ff\\",
     top_dir + "2022_ME36_nf_ff_sparsified\\metadata\\"]
elif 'loc' not in locals():
    raise NameError("add a 'loc' varaible to this script with the raw, " +
                    "reduced, and metadata folders")

source_dir, destination_dir, meta_dir = locs

#########################################
#  Get the list of tasks that need doing
#########################################
exp_names = ["me3-6-broke-nf-1",
             "me3-6-unloaded-nf-2"]

sources = [source_dir + n for n in exp_names if os.path.exists(source_dir + n)]
tasks = {}
for source_dir in sources:
    lr = LogReader(source_dir, destination_dir)
    tasks.update(lr.task_list)

# prune the task list
keys = [x for x in tasks.keys()]
for key in keys:
    # few tests were tiny scans to check on the part. ignore these.
    if tasks[key]['nframes'] < 10:
        tasks.pop(key)
        print(" -- {} too tiny to matter; tossing out".format(key))
    # If the raw data is missing on the local machine, toss it.
    elif not os.path.exists(tasks[key]['from']):
        tasks.pop(key)
        print(" -- {} cannot find raw data".format(key))
    # If overwrite is false and reduced data already exists, skip.
    elif len(glob.glob(tasks[key]['to'] + key)) > 0:
        if not overwrite:
            print(" -- {} already reduced. skipping".format(key))
            tasks.pop(key)
        else:
            print(" -- {} already reduced. overwriting".format(key))
keys = [x for x in tasks.keys()]

# if this is on OSC, allow parsible input so we can run one task per node
# Otherwise, assume we want to run on all files in the source location
if SLURM:
    id_to_run = int(sys.argv[1])
    id_count = len(tasks.keys())
    if id_to_run >= id_count:
        print("id {} is outside of range {}. Terminating".format(
            id_to_run + 1, id_count))
        sys.exit()
    print("Seting up Chore {} of {}".format(id_to_run + 1, id_count))
    keys = [keys[id_to_run]]


#########################################
#  grab other run-specific metadata
#########################################
# Grab the load at each step from pso logs
# grab the epoch and z height from the meta scalars.txt
for key in keys:
    fname = glob.glob(tasks[key]['from'][:-2]+"/pso/*log.txt")[0]
    with open(fname, 'r') as f:
        txt = [f.readline()[:-2] for x in range(10)]
        f.close()
    load_txt = [x.split(",")[0] for x in ",".join(txt).split("(N)=")[1:]]
    load = np.array(load_txt, dtype=float).mean()
    tasks[key]['load'] = load
    arr = np.loadtxt(tasks[key]['from'][:-2]+"/meta/1/scalars.txt")
    tasks[key]['epoch'] = np.mean(arr[:, 2])
    tasks[key]['z_height'] = np.mean(arr[:, 4])

# Sort keys into chronological order (makes it easier to spot mistakes)
task_names = np.array(keys)[np.argsort([tasks[x]['epoch'] for x in keys])]

# load up all the other weird metadata we will want.
det_red = Detector_Reducer()
print(meta_dir)
default_instr_location = glob.glob(meta_dir+"manta_semi_calibrated.yml")[0]
instr_dict = yaml.safe_load(open(default_instr_location,'r'))


#########################################
# Run the tasks
#########################################
# tasks are parallelized internally, so they should be ran in serial here.
# Code is I/O limited, so multiprocessing this is of limited use.
for i, task_name in enumerate(task_names):
    print("########")
    print("  Running {}".format(task_name))
    print("    (task {} of {})".format(i + 1, len(task_names)))
    print("########")
    tic = time.time()
    reduce_entire_nf(tasks[task_name], task_name, det_red, instr_dict)
    toc = time.time()-tic
    print("########")
    print("    {} done. total time: {} seconds".format(task_name, int(toc)))
    print("########")
