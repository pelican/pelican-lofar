include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# === Create library of pipelines.
set(pipeline_lib_src
    src/UdpBFPipeline.cpp
    src/UdpBFApplication.cpp
    src/EmbraceBFPipeline.cpp
    src/EmbraceBFApplication.cpp
    src/H5CVPipeline.cpp
    src/H5CVApplication.cpp
    src/TimingPipeline.cpp
    src/EmptyPipeline.cpp
    src/StokesOnlyPipeline.cpp
    src/SigProcPipeline.cpp
    src/BandPassPipeline.cpp
    src/JTestPipeline.cpp
)
if(CUDA_FOUND)
    list(APPEND pipeline_lib_src
         src/DedispersionPipeline.cpp
         src/DedispersionApplication.cpp
    )
endif(CUDA_FOUND)
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
if(CUDA_FOUND)
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
install(TARGETS dedispersionStream1 dedispersionStream2
		DESTINATION ${BINARY_INSTALL_DIR})
endif(CUDA_FOUND)

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
# === Build the EMBRACE Beamforming pipeline binary.
add_executable(embraceBFPipeline src/embraceBFmain.cpp)
set_target_properties(embraceBFPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(embraceBFPipeline
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)

# === Build the EMBRACE Beamforming pipeline binary for stream 1.
add_executable(embraceBFPipelineStream1 src/embraceBFmainStream1.cpp)
set_target_properties(embraceBFPipelineStream1 PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(embraceBFPipelineStream1
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)

# === Build the EMBRACE Beamforming pipeline binary for stream 2.
add_executable(embraceBFPipelineStream2 src/embraceBFmainStream2.cpp)
set_target_properties(embraceBFPipelineStream2 PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(embraceBFPipelineStream2
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)
# === Build the H5CV Beamforming pipeline binary.
add_executable(h5CVPipeline src/h5CVmain.cpp)
set_target_properties(h5CVPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(h5CVPipeline
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)

# === Build the H5CV Beamforming pipeline binary for stream 1.
add_executable(h5CVPipelineStream1 src/h5CVmainStream1.cpp)
set_target_properties(h5CVPipelineStream1 PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(h5CVPipelineStream1
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)

# === Build the H5CV Beamforming pipeline binary for stream 2.
add_executable(h5CVPipelineStream2 src/h5CVmainStream2.cpp)
set_target_properties(h5CVPipelineStream2 PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(h5CVPipelineStream2
    pelicanMdsm
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
)


# === Build the JTest pipeline binary
add_executable(JTestPipelineStream src/JTestPipelineMain.cpp)
set_target_properties(JTestPipelineStream PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(JTestPipelineStream
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
install(TARGETS emptyPipeline udpBFPipeline 
                udpBFPipelineStream1 udpBFPipelineStream2
		embraceBFPipeline 
                embraceBFPipelineStream1 embraceBFPipelineStream2 
		h5CVPipeline h5CVPipelineStream1
		h5CVPipelineStream2
                DESTINATION ${BINARY_INSTALL_DIR})

#install(TARGETS udpBFPipeline embraceBFPipeline DESTINATION ${BINARY_INSTALL_DIR})

include(CopyFiles)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.xml . mdsmXmlFiles)

add_subdirectory(test)
