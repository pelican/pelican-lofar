include_directories(${CMAKE_CURRENT_SOURCE_DIR} ..)

# Create test utility library.
# ------------------------------------------------------------------------------
set(lofarTestLib_src
    src/LofarUdpEmulator.cpp
    src/LofarEmulatorDataSim.cpp
)
add_library(lofarTestLib STATIC ${lofarTestLib_src})


# Create test binary and add it to the cmake test framework.
# ------------------------------------------------------------------------------
set(lofarTest_src
    src/libTest.cpp
    #src/LofarChunkerTest.cpp
    #src/DataStreamingTest.cpp
    ####src/PelicanBlobClientTest.cpp
    #src/PolyphaseCoefficientsTest.cpp
    src/AdapterTimeSeriesDataSetTest.cpp
    #src/SpectrumTest.cpp
    #src/SpectrumDataSetTest.cpp
    src/PPFChanneliserTest.cpp
)

add_executable(lofarTest ${lofarTest_src})
set_target_properties(lofarTest PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(lofarTest
    pelican-lofar_static
    lofarTestLib
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${CPPUNIT_LIBRARIES}
)

add_test(lofarTest lofarTest)


# Create the lofar udp emulator binary.
# ------------------------------------------------------------------------------
add_executable(lofarEmulator src/lofarEmulatorMain.cpp)
target_link_libraries(lofarEmulator
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    lofarTestLib
)

# Create the lofar binary which sends simulated data.
# ------------------------------------------------------------------------------
add_executable(lofarEmulatorDataSim src/lofarEmulatorDataSimMain.cpp)
target_link_libraries(lofarEmulatorDataSim
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    lofarTestLib
)



# Create a non cppunit PPF test binary
# ------------------------------------------------------------------------------
add_executable(ppfTest src/PPFChanneliserTestMain.cpp)
set_target_properties(ppfTest PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(ppfTest
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)


# Copy files required for testing to the build directory.
# ------------------------------------------------------------------------------
include(${CMAKE_SOURCE_DIR}/cmake/CopyFiles.cmake)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.* . testLibFiles)

