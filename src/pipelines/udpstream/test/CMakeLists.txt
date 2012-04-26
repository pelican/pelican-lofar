include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ..)

# Create test utility library.
# ------------------------------------------------------------------------------
set(pipelineTestLib_src
#    src/LofarEmulatorDataSim.cpp
)
#add_library(pipelineTestLib STATIC ${lofarTestLib_src})

set( TEST_DATA_DIR
     ${CMAKE_CURRENT_SOURCE_DIR}/data
   )
set(TEST_FLAGS
    -DTEST_DATA=\"${TEST_DATA_DIR}\"
   )


# Create test binary and add it to the cmake test framework.
# ------------------------------------------------------------------------------
set(pipelineTest_src
    src/pipelineTest.cpp
    src/LofarTestClient.cpp
    src/UdpBFPipelineIntegrationTest.cpp
    src/EmulatorPipeline.cpp
)
if(CUDA_FOUND)
    list(APPEND pipelineTest_src
         src/DedispersionPipelineTest.cpp
	)
endif(CUDA_FOUND)

set( pipelineTest_moc_headers
     LofarTestClient.h
   )

QT4_WRAP_CPP(pipelineTest_src ${pipelineTest_moc_headers})

add_executable(pipelineTest ${pipelineTest_src})
set_target_properties(pipelineTest PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS} ${TEST_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(pipelineTest
#    pelican-lofar_static
    pelican-lofar
    pelicanMdsm
#    pipelineTestLib
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

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/testInfo.in ${CMAKE_CURRENT_BINARY_DIR}/testInfo.h)

add_test(pipelineTest pipelineTest)


# Copy files required for testing to the build directory.
# ------------------------------------------------------------------------------
#include(${CMAKE_SOURCE_DIR}/cmake/CopyFiles.cmake)
#copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.* . testLibFiles)

