#
# pelican-lofar/lib/test/CMakeLists.txt
#

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
    src/BinMapTest.cpp
    src/BandPassTest.cpp
    src/RFI_ClipperTest.cpp
    src/PumaOutputTest.cpp
    src/DataStreamingTest.cpp
    ##src/FilterBankAdapterTest.cpp
    ###src/PelicanBlobClientTest.cpp
    src/PolyphaseCoefficientsTest.cpp
    src/AdapterTimeSeriesDataSetTest.cpp
    src/SpectrumDataSetTest.cpp
    src/PPFChanneliserTest.cpp
    src/LofarChunkerTest.cpp
    src/LofarDataSplittingChunkerTest.cpp
)

add_executable(lofarTest ${lofarTest_src})
set_target_properties(lofarTest PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(lofarTest
    pelican-lofar_static
    lofarTestLib
    ${PELICAN_TESTUTILS_LIBRARY}
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${CPPUNIT_LIBRARIES}
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
)

add_test(lofarTest lofarTest)


# Create the lofar udp emulator binary.
# ------------------------------------------------------------------------------
add_executable(lofarEmulator src/lofarEmulatorMain.cpp)
target_link_libraries(lofarEmulator
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    lofarTestLib
)

# Create the lofar binary which sends simulated data.
# ------------------------------------------------------------------------------
add_executable(lofarEmulatorDataSim src/lofarEmulatorDataSimMain.cpp)
target_link_libraries(lofarEmulatorDataSim
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    lofarTestLib
)
install(TARGETS "lofarEmulatorDataSim" DESTINATION ${BINARY_INSTALL_DIR})



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
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
)


# Test the channel profile of the PPF channeliser.
# -----------------------------------------------------------------------------
add_executable(ppf_channel_profile src/PPF_ChannelProfileTest.cpp)
set_target_properties(ppf_channel_profile PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(ppf_channel_profile
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
)




# Copy files required for testing to the build directory.
# ------------------------------------------------------------------------------
include(${CMAKE_SOURCE_DIR}/cmake/CopyFiles.cmake)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.* . testLibFiles)

