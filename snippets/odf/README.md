# **Checkers/ODF**

Place to share code for building an ODF tool to use with HEXRD/ORIX/Diffraction data in general.

## **MTEX_ODF**
The original paper that MTEX is based on is actually a Pole Figure Inversion tool ([link to paper here](https://www-user.tu-chemnitz.de/~rahi/paper/mtex_paper.pdf)). The paper does a great job of laying out what it does and how. My favorite line from the whole thing:

> "We render an explicit definition of harmonics as there are
many slightly different ways to define them, e.g. with respect to
normalization, the disastrous impact of which is only revealed
in the course of writing and checking software code. There-
fore, it is our hope that the reader and practitioner of texture
analysis and open-source software development will
appreciate a comprehensive and consistent view of a method
with unique features."

Beautiful.

However, the work itself is non-trivial, and despite publishing both his code and a step-by-step how to guide, no one has ever redone the inversion in python.  

To this end, I need to create a pythonic equivalent to the executable "pf2odf" that was in MTEX up until version 4.0, where it was removed in favor of precompiled .mex files. 

## **MTEX_ODF/C**
The files in here are initialized as a near-identical clone to Ralph's MTEX 4.0 c code, then edited to remove the unnecessary .mex and Matlab builders, at -g flags, and other changes. Used primarily for running through gdb to see step-by-step changes to code.

MTEX uses both FFTW and NFFT to compile. NFFT changed the names of their classes a few times between 2004 and 2014 (ie MTEX 0.1 amd MTEX 4.0), so I suggest always builting from the 2014 version to avoid linking errors, which uses NFFT3.2.3

Magic Command to compile FFTW-3.3.10 with everything necessary for NFFT3 and MTEX:

> ./configure --prefix=/local/scratch/agerlt/libs  --enable-shared --enable-threads --enable-openmp
> 
> make
> 
> make install

Magic command to compile NFFT3.2.3 with everything necessary for MTEX 4.0:
> ./bootstrap.sh
./configure --enable-all --enable-openmp --with-matlab=/opt/local/matlab/R2021a --prefix=/local/scratch/agerlt/libs --with-fftw3-libdir=/local/scratch/agerlt/libs/lib --with-fftw3-includedir=/local/scratch/agerlt/libs/include --prefix=/local/scratch/agerlt/libs
> 
> make
> 
> make install
 

## **MTEX_ODF/dubna_input_data**
Copy of some text files that can be ran through pf2odf to see how it works. command is "c/bin/pf2odf dubna_input_data/pf2odf5666.txt"

up until 2015-ish, MTEX ran by using Matlab code to write binary and text files to disk, then process them with c executables, then have MTEX read in the results. Modern MTEX uses .mex files to just point to memory addresses. THe Modern solution is better, but the older version is easier to read/follow, hence the use of 2014 code. You can get the equivalent of this file in both MTEX 4.0 and modern 5.7 by running the "Dubna" example. 5.7 passes variables into a .mex, whereas 4.0 writes to the "tmp" folder
