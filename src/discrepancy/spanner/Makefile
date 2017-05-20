####### Compiler, tools and options

CC            = gcc
CXX           = g++
CFLAGS        = -O2 -Wall -W -fPIC
CXXFLAGS      = -O2 -Wall -W -fPIC -std=c++0x#-std=c++11
INCPATH       = -I.
LINK          = g++
LFLAGS        = -O2 -shared
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f

####### Output directory
OBJECTS_DIR   = ./

####### Files
SOURCES       = api.cpp
OBJECTS       = api.o
TARGET        = libstein_spanner.so
TARGET0       = libstein_spanner.1.0.0.so
TARGET1       = libstein_spanner.1.so
TARGET2       = libstein_spanner.1.0.so

####### Implicit rules
.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

####### Build rules
all: Makefile  $(TARGET)

$(TARGET):  $(OBJECTS)
	-$(DEL_FILE) $(TARGET) $(TARGET0) $(TARGET1) $(TARGET2)
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS)
	-ln -s $(TARGET) $(TARGET0)
	-ln -s $(TARGET) $(TARGET1)
	-ln -s $(TARGET) $(TARGET2)

clean:
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core

####### Sub-libraries
distclean: clean
	-$(DEL_FILE) $(TARGET)
	-$(DEL_FILE) $(TARGET0) $(TARGET1) $(TARGET2)

####### Compile
api.o: api.cpp greedyspanner.h \
		structures.h \
		binaryheap.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o api.o api.cpp
