include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(lib_src
    src/LofarChunker.cpp
    src/LofarStreamDataClient.cpp
    src/LofarServerClient.cpp
    src/ChannelisedStreamData.cpp
    src/AdapterTimeStream.cpp
    src/ChanneliserPolyphase.cpp
    src/PolyphaseCoefficients.cpp
    src/PelicanBlobClient.cpp
)
add_library(pelicanLofar ${lib_src} )

add_executable(blobClient blobClientMain.cpp)
target_link_libraries(blobClient pelicanLofar)

set(SUBPACKAGE_LIBRARIES
    pelicanLofar
)

# Link with openmp.
set_target_properties(pelicanLofar PROPERTIES 
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(pelicanLofar ${PROJECT_LIBRARIES})
install(TARGETS pelicanLofar DESTINATION ${LIBRARY_INSTALL_DIR})

add_subdirectory(test)
