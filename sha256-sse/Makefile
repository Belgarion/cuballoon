
all: test profile

CC = gcc
CFLAGS = -g -O3 -no-pie

OBJS = sha256-ref.o sha256-sse.o

%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

test: $(OBJS) test.o 
	$(CC) $(CFLAGS) -o $@ $(OBJS) test.o

profile: $(OBJS) profile.o
	$(CC) $(CFLAGS) -o $@ $(OBJS) profile.o

clean:
	-rm -f $(OBJS) test
