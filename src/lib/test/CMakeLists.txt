#
# pelican-lofar/lib/test/CMakeLists.txt
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ==== Create test utility library.
set(lofarTestLib_src
    src/LofarUdpEmulator.cpp
    src/LofarEmulatorDataSim.cpp
    src/GPU_TestCard.cpp
    src/DedispersionDataGenerator.cpp
    src/LofarPipelineTester.cpp
    src/LofarDataBlobGenerator.cpp
)

# CUDA routines
set(cuda_src
   CudaVectorAdd.cu
   )
if( CUDA_FOUND )
    CUDA_COMPILE( cuda_obj ${cuda_src} )
    list(APPEND lofarTestLib_src 
         src/TestCudaVectorAdd.cpp
        )
endif( CUDA_FOUND )

add_library(lofarTestLib STATIC ${lofarTestLib_src} ${cuda_obj} )
target_link_libraries(lofarTestLib
    pelican-lofar
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

# ==== Create test binary and add it to the cmake test framework.
set(lofarTest_src
    src/CppUnitMain.cpp
    src/GPU_ManagerTest.cpp
    src/GPU_MemoryMapTest.cpp
    src/AdapterTimeSeriesDataSetTest.cpp
    src/BandPassTest.cpp
    src/BinMapTest.cpp
    src/DataStreamingTest.cpp
    src/DedispersionAnalyserTest.cpp
    #src/FilterBankAdapterTest.cpp
    #src/LofarChunkerTest.cpp
    src/LockingContainerTest.cpp;
    src/LofarDataSplittingChunkerTest.cpp
    #src/PelicanBlobClientTest.cpp
    src/PPF_ChanneliserTest.cpp
    src/PPF_CoefficientsTest.cpp
    src/PumaOutputTest.cpp
    #src/RFI_ClipperTest.cpp
    src/SpectrumDataSetTest.cpp
)
if(CUDA_FOUND)
    list(APPEND lofarTest_src
            src/GPU_NVidiaTest.cpp
            src/GPU_ParamTest.cpp
            src/DedispersionBufferTest.cpp
            src/DedispersionModuleTest.cpp
        )
endif(CUDA_FOUND)

add_executable(lofarTest ${lofarTest_src})
set_target_properties(lofarTest PROPERTIES 
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
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
    ${QT_QTXML_LIBRARY})
add_test(lofarTest lofarTest)


# ==== Create the lofar udp emulator binary.
add_executable(lofarEmulator src/lofarEmulatorMain.cpp)
target_link_libraries(lofarEmulator
    ${PELICAN_LIBRARY}
    pelican-lofar_static
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${QT_QTXML_LIBRARY}
    lofarTestLib
)

# ==== Create the lofar binary which sends simulated data.
add_executable(lofarEmulatorDataSim src/lofarEmulatorDataSimMain.cpp)
target_link_libraries(lofarEmulatorDataSim
    ${PELICAN_LIBRARY}
    pelican-lofar_static
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    lofarTestLib
)
install(TARGETS "lofarEmulatorDataSim" DESTINATION ${BINARY_INSTALL_DIR})

# ==== Create a non cppunit PPF test binary
add_executable(ppf_performance_test src/PPF_PerformanceTest.cpp)
set_target_properties(ppf_performance_test PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(ppf_performance_test
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
)

# ==== Test the channel profile of the PPF channeliser.
add_executable(ppf_profile_test src/PPF_ProfileTest.cpp)
set_target_properties(ppf_profile_test PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(ppf_profile_test
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
)

# ==== Copy files required for testing to the build directory.
include(${CMAKE_SOURCE_DIR}/cmake/CopyFiles.cmake)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.* . testLibFiles)
