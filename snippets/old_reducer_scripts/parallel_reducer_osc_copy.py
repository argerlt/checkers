import numpy as np
import glob
import time
from pr_functions import LogReader
from pr_functions import Detector_Reducer
from pr_functions import reduce_entire_ff

parent_raws = "/local/scratch/2022_me36_ff_sparse/"
top_red = "/local/scratch/2022_me36_ff_sparse/data"

lr = LogReader(parent_raws+"me3-6_fatigue", top_red)
tasks = lr.task_list

# few tests were tiny scans to see if the part was still there. ignore these.
keys = [x for x in tasks.keys()]
for key in keys:
    if tasks[key]['nframes'] < 1440:
        tasks.pop(key)
        print(" -- {} too tiny to matter; tossing out".format(key))

# I want the loads at each step too, so we grab that from the pso log.s
# grab the epoch and z height as well from the meta scalars.txt
keys = [x for x in tasks.keys()]
keys = [keys[4]]
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

# finally, sort keys into chronological order
chore_names = np.array(keys)[np.argsort([tasks[x]['epoch'] for x in keys])]


x = chore_names[0]
det_red = Detector_Reducer()
print(tasks[x])
print("")
print("")
print(chore_names)
print("")
print("")
tic = time.time()
reduce_entire_ff(tasks[x], x, det_red, 5)
toc = time.time()-tic
print(toc)
