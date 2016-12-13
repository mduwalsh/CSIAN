CC = gcc
LIBS = -lm
CFLAGS = -Wall -O2 -march=native

EXECUTABLES = csi 

.PHONY: all clean

all: $(EXECUTABLES)

csi: csi.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o $(EXECUTABLES)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@
