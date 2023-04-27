# Chopped down version of MTEX Makefile
#
#--------------- begin editable section -------------------------------
#
# Define OS (just in case we compile for more than just linus)
# OSU_Linux - 64 bit linuxglnx86  - 32 bit Linux
# win64   - 64 bit Windows
#
TARGET= glnxa64
RNAME = mtex-4.0.23
#
# Path NFFT and FFTW headers, ie nfft.h and fftw3.h
H_PATH = /local/scratch/agerlt/libs/include
# path to NFFT and FFTW libraries, i.e. to libnfft3.a and libfftw3.a
LIB_PATH = /local/scratch/agerlt/libs/lib/
# Additional Flags
CFLAGS= -c -g -O3 -Wall -fomit-frame-pointer -fstrict-aliasing -ffast-math -mfpmath=sse,387 -mieee-fp -m3dnow -mmmx -msse -msse2
LDFLAGS= -lm
#--------------- end editable section ---------------------------------
#
# local directories
BPATH = ./
SUBDIRS = kernel.dir #test.dir


# top-level rule, to compile everything.
all: $(SUBDIRS)


# descent into subdirectories
%.dir:
	$(MAKE) -e CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" LIB_PATH=$(LIB_PATH) H_PATH=$(H_PATH) TARGET="$(TARGET)" -C $*
	$(MAKE) TARGET="$(TARGET)" install -C $*


# rule for cleaning re-compilable files.
clean:
# rm -f c/bin/*
	find . -name '*~' -or -name '*.log' -or -name '.directory' -or -name '*.o' -or -name '*.mex*' | xargs /bin/rm -rf


