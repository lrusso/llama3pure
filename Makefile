# choose your compiler, e.g. gcc/clang
# example override to clang: make llama3pure CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: llama3pure
llama3pure: llama3pure-c-engine.c
	$(CC) -O3 -o llama3pure llama3pure-c-engine.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./llama3pure model.gguf 1.0 3
llama3puredebug: llama3pure-c-engine.c
	$(CC) -g -o llama3pure llama3pure-c-engine.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: llama3purefast
llama3purefast: llama3pure-c-engine.c
	$(CC) -Ofast -o llama3pure llama3pure-c-engine.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./llama3pure model.gguf
.PHONY: llama3pureomp
llama3pureomp: llama3pure-c-engine.c
	$(CC) -Ofast -fopenmp -march=native llama3pure-c-engine.c -lm -o llama3pure

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc-win32 -Ofast -D_WIN32 -o llama3pure.exe -I. llama3pure-c-engine.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: llama3puregnu
llama3puregnu:
	$(CC) -Ofast -std=gnu11 -o llama3pure llama3pure-c-engine.c -lm

.PHONY: llama3pureompgnu
llama3pureompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 llama3pure-c-engine.c -lm -o llama3pure

.PHONY: clean
clean:
	rm -f llama3pure llama3pure.exe
