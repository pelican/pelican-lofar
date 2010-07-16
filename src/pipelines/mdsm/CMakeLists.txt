include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# === Create library of pipelines.
set(pipeline_lib_src
    src/MdsmPipeline.cpp
    src/UdpBFPipeline.cpp
)
add_library(pelicanMdsm ${pipeline_lib_src})

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

#install(TARGETS udpBFPipeline DESTINATION ${BINARY_INSTALL_DIR})


# === OLD pipeline
add_executable(mdsmPipeline src/main.cpp)
set_target_properties(mdsmPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(mdsmPipeline
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
)

install(TARGETS mdsmPipeline DESTINATION ${BINARY_INSTALL_DIR})

include(CopyFiles)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.xml . mdsmXmlFiles)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.dat . mdsmChanCoeffs)

#add_subdirectory(test)
