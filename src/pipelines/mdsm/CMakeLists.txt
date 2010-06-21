include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(pipeline_lib_src
    src/MdsmPipeline.cpp
)
add_library(pelicanMdsm ${pipeline_lib_src} )

add_executable(mdsmPipeline src/main.cpp)

target_link_libraries(mdsmPipeline pelicanMdsm pelicanLofar ${FFTW3_LIBRARIES})
install(TARGETS mdsmPipeline DESTINATION ${BINARY_INSTALL_DIR})

#add_subdirectory(test)
