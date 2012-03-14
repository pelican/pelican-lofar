include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# === Create library of pipelines.
set(pipeline_lib_src
    src/UdpBFPipeline.cpp
    src/UdpBFApplication.cpp
    src/TimingPipeline.cpp
    src/EmptyPipeline.cpp
    src/StokesOnlyPipeline.cpp
    src/SigProcPipeline.cpp
    src/BandPassPipeline.cpp
    src/DedispersionPipeline.cpp
    src/DedispersionApplication.cpp
)
add_library(pelicanMdsm ${pipeline_lib_src})

# === Build the Empty Pipeline for max performance testing
add_executable(emptyPipeline src/emptyPipelineMain.cpp)
target_link_libraries(emptyPipeline
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)

# === Build the dedispersion pipeline binary
add_executable(dedispersionStream1 src/dedispersionStream1.cpp)
set_target_properties(dedispersionStream1 PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(dedispersionStream1
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)
add_executable(dedispersionStream2 src/dedispersionStream2.cpp)
set_target_properties(dedispersionStream2 PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(dedispersionStream2
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
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
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
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
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
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
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)


# # === Build the sigprocPipeline
# add_executable(sigprocPipeline src/sigprocPipelineMain.cpp)
# set_target_properties(sigprocPipeline PROPERTIES
#     COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
#     LINK_FLAGS "${OpenMP_CXX_FLAGS}"
# )
# target_link_libraries(sigprocPipeline
#     pelicanMdsm
#     pelican-lofar_static
#     ${PELICAN_LIBRARY}
#     ${FFTW3_FFTW_LIBRARY}
#     ${FFTW3_FFTWF_LIBRARY}
#     ${QT_QTNETWORK_LIBRARY}
#     ${QT_QTXML_LIBRARY}
#     ${QT_QTCORE_LIBRARY}
# )

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
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)
install(TARGETS emptyPipeline udpBFPipeline udpBFPipelineStream1 udpBFPipelineStream2 DESTINATION ${BINARY_INSTALL_DIR})

#install(TARGETS udpBFPipeline DESTINATION ${BINARY_INSTALL_DIR})

include(CopyFiles)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.xml . mdsmXmlFiles)

add_subdirectory(test)
