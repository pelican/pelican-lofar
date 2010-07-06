include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create the pelican-lofar library and set its install target.
# ------------------------------------------------------------------------------
set(lib_src
    src/AdapterTimeStream.cpp
    src/ChannelisedStreamData.cpp
    src/ChanneliserPolyphase.cpp
    src/LofarChunker.cpp
    src/LofarServerClient.cpp
    src/LofarStreamDataClient.cpp
    src/LofarStationConfiguration.cpp
    src/LofarStationConfigurationAdapter.cpp
    src/PelicanBlobClient.cpp
    src/PolyphaseCoefficients.cpp
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
target_link_libraries(pelican-lofar
    ${FFTW3_LIBRARIES}
)


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
