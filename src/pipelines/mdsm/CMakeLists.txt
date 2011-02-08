include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# === Create library of pipelines.
set(pipeline_lib_src
    src/UdpBFPipeline.cpp
    src/UdpBFPipelineStream1.cpp
    src/UdpBFPipelineStream2.cpp
    src/TimingPipeline.cpp
    src/EmptyPipeline.cpp
)
add_library(pelicanMdsm ${pipeline_lib_src})

# === Build the Empty Pipeline for max performance testing
add_executable(emptyPipeline src/emptyPipelineMain.cpp)
target_link_libraries(emptyPipeline
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
)

# === Build the UDP Beamforming pipeline binary.
add_executable(udpBFPipeline src/udpBFmain.cpp)
set_target_properties(udpBFPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(udpBFPipeline
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
)

# === Build the UDP Beamforming pipeline binary for stream 1.
add_executable(udpBFPipelineStream1 src/udpBFmainStream1.cpp)
set_target_properties(udpBFPipelineStream1 PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(udpBFPipelineStream1
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
)

# === Build the UDP Beamforming pipeline binary for stream 2.
add_executable(udpBFPipelineStream2 src/udpBFmainStream2.cpp)
set_target_properties(udpBFPipelineStream2 PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(udpBFPipelineStream2
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
)


# === Build the Timing pipeline binary.
add_executable(timingPipeline src/timingPipelineMain.cpp)
set_target_properties(timingPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(timingPipeline
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
)
install(TARGETS emptyPipeline udpBFPipeline udpBFPipelineStream1 udpBFPipelineStream2 DESTINATION ${BINARY_INSTALL_DIR})

#install(TARGETS udpBFPipeline DESTINATION ${BINARY_INSTALL_DIR})

include(CopyFiles)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.xml . mdsmXmlFiles)

#add_subdirectory(test)
