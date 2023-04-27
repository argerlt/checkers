# Chopped down version of MTEX kernel Makefile
#
#--------------- begin editable section -------------------------------
#
# Path NFFT and FFTW headers, ie nfft.h and fftw3.h
H_PATH ?= /local/scratch/agerlt/libs/include
# path to NFFT and FFTW libraries, i.e. to libnfft3.a and libfftw3.a
LIB_PATH ?= /local/scratch/agerlt/libs/lib/
# Additional Flags
CFLAGS ?= -c -g -O3 -Wall -fomit-frame-pointer -fstrict-aliasing -ffast-math -mfpmath=sse,387 -mieee-fp -m3dnow -mmmx -msse -msse2

#--------------- end editable section ---------------------------------
#
# local directories
BPATH ?= ../


CC=gcc
MYCFLAGS=$(CFLAGS) -I$(H_PATH) -I../include
LDFLAGS ?= -lm #-lpthread
TPATH = ../tools

# list of static libraris
LIBS = $(LIB_PATH)/libnfft3.a $(LIB_PATH)/libfftw3.a #$(FFTW_LIB_PATH)/libfftw3_threads.a

# list of generated object files.
IOTOOLS = $(TPATH)/pio.o $(TPATH)/helper.o $(TPATH)/sparse.o
SOLVERTOOLS = zbfm.o zbfm_solver.o odf.o

#
TARGETS = odf2pf pf2odf pdf2pf fc2odf odf2fc #pf2pdf

# top-level rule, to compile everything.
all: $(TARGETS)

# odf -> pdf
odf2pf: odf2pf.o pdf.o $(IOTOOLS)
	$(CC) $(IOTOOLS) pdf.o $(LIBS) $@.o -o $@ $(LDFLAGS)

# pdf -> odf
pf2odf: pf2odf.o pdf.o $(IOTOOLS) $(SOLVERTOOLS)
	$(CC) $(IOTOOLS) pdf.o $(SOLVERTOOLS) $(LIBS) $@.o -o $@ $(LDFLAGS)

# pdf -> Fourier coefficients
pf2pdf: pf2pdf.o $(IOTOOLS) pdf.o ipdf.o
	$(CC) $(IOTOOLS) pdf.o ipdf.o $(LIBS) $@.o -o $@ $(LDFLAGS)

# Fourier coefficients -> pdf
pdf2pf: pdf2pf.o $(IOTOOLS) pdf.o
	$(CC) $(IOTOOLS) pdf.o $(LIBS) $@.o -o $@ $(LDFLAGS)

# Fourier coefficients - odf
fc2odf: fc.o fc2odf.o $(IOTOOLS)
	$(CC) $(IOTOOLS) fc.o $@.o $(LIBS) -o $@ $(LDFLAGS)

# odf -> Fourier coefficients
odf2fc: fc.o odf2fc.o $(IOTOOLS)
	$(CC) $(IOTOOLS) fc.o $@.o $(LIBS) -o $@ $(LDFLAGS)

install:
	mv $(TARGETS) $(BPATH)

%.o: %.c
	$(CC) $(MYCFLAGS) -c $< -o $@

clean:
	rm -f ../bin/*
	find . -name '*~' -or -name '*.log' -or -name '.directory' -or -name '*.o' -or -name '*.mex*' | xargs /bin/rm -rf