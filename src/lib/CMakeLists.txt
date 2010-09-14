include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create the pelican-lofar library and set its install target.
# ------------------------------------------------------------------------------
set(lib_src
    src/AdapterTimeSeriesDataSet.cpp
    src/OutputHDF5Lofar.cpp
    src/LofarData.cpp
    src/LofarChunker.cpp
    src/LofarServerClient.cpp
    src/LofarStreamDataClient.cpp
    src/LofarStationConfiguration.cpp
    src/LofarStationConfigurationAdapter.cpp
    src/PelicanBlobClient.cpp
    src/PumaOutput.cpp
    src/PolyphaseCoefficients.cpp
    src/SpectrumDataSet.cpp
    src/TimeSeriesDataSet.cpp
    src/PPFChanneliser.cpp
    src/StokesGenerator.cpp
    src/StokesIntegrator.cpp
    src/SigprocStokesWriter.cpp
)

# Create pelican-lofar library.
add_library(pelican-lofar_static STATIC ${lib_src})
set_target_properties(pelican-lofar_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(pelican-lofar_static PROPERTIES OUTPUT_NAME pelican-lofar)
set_target_properties(pelican-lofar_static PROPERTIES PREFIX lib)
set_target_properties(pelican-lofar_static PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
install(TARGETS pelican-lofar_static DESTINATION ${LIBRARY_INSTALL_DIR})

add_library(pelican-lofar SHARED ${lib_src})
set_target_properties(pelican-lofar PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(pelican-lofar PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
install(TARGETS pelican-lofar DESTINATION ${LIBRARY_INSTALL_DIR})

set(pelican-lofar_externals
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    #${CBLAS_LIBS}
)

target_link_libraries(pelican-lofar_static ${pelican-lofar_externals})

target_link_libraries(pelican-lofar ${pelican-lofar_externals})


file(GLOB public_headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
install(FILES ${public_headers} DESTINATION "${INCLUDE_INSTALL_DIR}")


# Create the blob client binary.
# ------------------------------------------------------------------------------
add_executable(blobClient blobClientMain.cpp)
target_link_libraries(blobClient
    ${PELICAN_LIBRARY}
    pelican-lofar_static
)

# Recurse into the test directory.
add_subdirectory(test)
