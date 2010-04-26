include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set( lib_src
    src/LofarChunker.cpp
    src/LofarStreamDataClient.cpp
    src/LofarServerClient.cpp
#   src/AdapterTimeStream.cpp
)
add_library(pelicanLofar ${lib_src} )

set(SUBPACKAGE_LIBRARIES
    pelicanLofar
)
target_link_libraries(pelicanLofar ${PROJECT_LIBRARIES} )
install(TARGETS pelicanLofar DESTINATION ${LIBRARY_INSTALL_DIR} )

add_subdirectory(test)
