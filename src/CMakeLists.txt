cmake_minimum_required(VERSION 2.4)

# === CMake Build File
project(pelican-lofar CXX C)

# === Project version
set(${PROJECT_NAME}_MAJOR_VERSION 0)
set(${PROJECT_NAME}_MINOR_VERSION 0)
set(${PROJECT_NAME}_PATCH_LEVEL 0)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)
enable_testing()

# === Options and defaults
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug)
endif (NOT CMAKE_BUILD_TYPE)

if (NOT PELICAN_INSTALL_DIR)
    set(PELICAN_INSTALL_DIR "pelican")
endif (NOT PELICAN_INSTALL_DIR )

if (NOT LIBRARY_INSTALL_DIR)
    set(LIBRARY_INSTALL_DIR ${PELICAN_INSTALL_DIR}/lib)
endif (NOT LIBRARY_INSTALL_DIR)

if (NOT BINARY_INSTALL_DIR)
    set(BINARY_INSTALL_DIR ${PELICAN_INSTALL_DIR}/bin)
endif (NOT BINARY_INSTALL_DIR)

if (NOT DOC_INSTALL_DIR)
    set(DOC_INSTALL_DIR ${PELICAN_INSTALL_DIR}/doc)
endif (NOT DOC_INSTALL_DIR)

# === Find Dependencies
find_package(Pelican REQUIRED)
find_package(CppUnit REQUIRED)

#list(APPEND CMAKE_CXX_FLAGS "-pg")

# === Add directories to the include path (i.e. -I<directory> ...)
include_directories(
    ${PELICAN_INCLUDE_DIR}
    ${PELICAN_INCLUDES}
)

set(PROJECT_LIBRARIES
    ${PELICAN_LIBRARIES}
)
link_directories(${PELICAN_LIBRARIES})
list(INSERT PROJECT_LIBRARIES 0
    pelicanLofar
)

# == Recurse into sub-directories defined in modules.cmake.
include(modules.cmake)


# === Create the stream client binary.
add_executable(lofarClient
    lofarClient.cpp
    # pipeline.cpp
)
target_link_libraries(lofarClient ${PROJECT_LIBRARIES})


# === Create the server binary.
#set( serverBinSrc
#     src/serverBin.cpp
#)

#add_executable( lofarServerClient ${serverClientBinSrc} )
#add_executable( lofarServer ${serverBinSrc} )
