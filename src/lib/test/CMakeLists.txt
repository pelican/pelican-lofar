include_directories(${CMAKE_CURRENT_SOURCE_DIR} ..)

# Create test utility library.
# ------------------------------------------------------------------------------
set(lofarTestLib_src
    src/LofarUdpEmulator.cpp
)
add_library(lofarTestLib STATIC ${lofarTestLib_src})


# Create test binary and add it to the cmake test framework.
# ------------------------------------------------------------------------------
set(lofarTest_src
    src/libTest.cpp
    src/LofarChunkerTest.cpp
    src/DataStreamingTest.cpp
    src/TimeStreamDataTest.cpp
    src/ChannelisedStreamDataTest.cpp
    src/AdapterTimeStreamTest.cpp
    src/PolyphaseCoefficientsTest.cpp
    src/ChanneliserPolyphaseTest.cpp
#    src/PelicanBlobClientTest.cpp
)

add_executable(lofarTest ${lofarTest_src})
set_target_properties(lofarTest PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(lofarTest
    lofarTestLib
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_LIBRARIES}
    ${CPPUNIT_LIBRARIES}
)
add_test(lofarTest lofarTest)


# Create the lofar udp emulator binary.
# ------------------------------------------------------------------------------
add_executable(lofarEmulator
    src/lofarEmulatorMain.cpp
)
target_link_libraries(lofarEmulator
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    lofarTestLib
)

# Copy files required for testing to the build directory.
include(CopyFiles)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.* . testLibFiles)


