include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# === Create the pelican-lofar library and set its install target.
set(lib_src
    src/AdapterTimeSeriesDataSet.cpp
    src/BinMap.cpp
    src/BandPassAdapter.cpp
    src/BandPass.cpp
    src/BandPassOutput.cpp
    src/BandPassRecorder.cpp
    src/BlobStatistics.cpp
    src/DedispersionBuffer.cpp
    src/EmbraceChunker.cpp
    src/FilterBankHeader.cpp
    src/FilterBankAdapter.cpp
    src/OutputHDF5Lofar.cpp
    src/GPU_Job.cpp
    src/GPU_Resource.cpp
    src/GPU_Manager.cpp
    src/LofarData.cpp
    src/LofarChunker.cpp
    src/LofarPelicanClientApp.cpp
    src/LofarServerClient.cpp
    src/LofarStreamDataClient.cpp
    src/LofarStationConfiguration.cpp
    src/LofarStationConfigurationAdapter.cpp
    src/PelicanBlobClient.cpp
    src/PumaOutput.cpp
    src/PolyphaseCoefficients.cpp
    src/RFI_Clipper.cpp
    src/RTMS_Data.cpp
    src/SpectrumDataSet.cpp
    src/TimeSeriesDataSet.cpp
    src/PPFChanneliser.cpp
    src/StokesGenerator.cpp
    src/StokesIntegrator.cpp
    src/SigprocStokesWriter.cpp
    src/LofarDataSplittingChunker.cpp
    src/WeightedSpectrumDataSet.cpp
    src/AsyncronousTask.cpp
    src/GPU_Task.cpp
    src/GPU_MemoryMap.cpp
)
if(CUDA_FOUND)
    list(APPEND lib_src
            src/AsyncronousModule.cpp
            src/DedispersionModule.cpp
            src/GPU_NVidia.cpp
            src/GPU_NVidiaConfiguration.cpp
            src/GPU_Kernel.cpp
        )
endif(CUDA_FOUND)

set(lib_moc_headers
    AsyncronousTask.h
    GPU_Task.h
)
QT4_WRAP_CPP(moc_src ${lib_moc_headers})

set(lib_cuda_src
    DedispersionKernel.cu
   )
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    CUDA_ADD_LIBRARY(ampp-cuda STATIC ${lib_cuda_src})
   set(cuda_libs ampp-cuda ${CUDA_LIBRARIES} )
#    set(cuda_libs ${CUDA_LIBRARIES} )
else(CUDA_FOUND)
    set(cuda_libs)
endif(CUDA_FOUND)

# === Create pelican-lofar library.
add_library(pelican-lofar_static STATIC ${moc_src} ${lib_src} )
set_target_properties(pelican-lofar_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(pelican-lofar_static PROPERTIES OUTPUT_NAME pelican-lofar)
set_target_properties(pelican-lofar_static PROPERTIES PREFIX lib)
set_target_properties(pelican-lofar_static PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
install(TARGETS pelican-lofar_static DESTINATION ${LIBRARY_INSTALL_DIR})

add_library(pelican-lofar SHARED ${moc_src} ${lib_src})
set_target_properties(pelican-lofar PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(pelican-lofar PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
install(TARGETS pelican-lofar DESTINATION ${LIBRARY_INSTALL_DIR})

set(pelican-lofar_externals
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${LAPACK_LIBRARIES}
    ${cuda_libs}
)

target_link_libraries(pelican-lofar_static ${pelican-lofar_externals})

target_link_libraries(pelican-lofar ${pelican-lofar_externals})


file(GLOB public_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
install(FILES ${public_headers} DESTINATION "${INCLUDE_INSTALL_DIR}")

# === Create the blob client binary.
add_executable(blobClient blobClientMain.cpp)
target_link_libraries(blobClient
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)

# Recurse into the test directory.
add_subdirectory(test)
