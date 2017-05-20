TEMPLATE = lib
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += api.cpp

HEADERS += \
    structures.h \
    greedyspanner.h \
    binaryheap.h

QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -fPIC
QMAKE_CXXFLAGS += -shared
