include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(pipeline_lib_src
    src/MdsmPipeline.cpp
)
add_library(pelicanMdsm ${pipeline_lib_src} )

add_executable(mdsmPipeline src/main.cpp)

# Link with openmp.
set_target_properties(mdsmPipeline PROPERTIES 
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(mdsmPipeline pelicanMdsm pelicanLofar ${FFTW3_LIBRARIES})
install(TARGETS mdsmPipeline DESTINATION ${BINARY_INSTALL_DIR})

#add_subdirectory(test)

include(CopyFiles)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.xml . mdsmXmlFiles)
