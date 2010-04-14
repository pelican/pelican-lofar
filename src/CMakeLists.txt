cmake_minimum_required(VERSION 2.4)

# === Set the compiler
if (INTEL)
  set(CMAKE_CXX_COMPILER icpc)
  set(CMAKE_C_COMPILER icc)
endif (INTEL)

# === CMake Build File
project(pelican-lofar CXX C)

# === Project version
set( ${PROJECT_NAME}_MAJOR_VERSION 0 )
set( ${PROJECT_NAME}_MINOR_VERSION 0 )
set( ${PROJECT_NAME}_PATCH_LEVEL 0 )

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

# === Dependencies
find_package(Pelican REQUIRED)
find_package(CppUnit)
include_directories(
    ${PELICAN_INCLUDE_DIR}
)

set(PROJECT_LIBRARIES
    ${PELICAN_LIBRARIES}
)

link_directories(${PELICAN_LIBRARIES})
list(INSERT PROJECT_LIBRARIES 0 
    pelicanLofar
    )

include(modules.cmake)

# === Binaries
#set( serverBinSrc
#     src/serverBin.cpp
#   )

set( streamClientBinSrc
     lofarClient.cpp
   )

add_executable( lofarClient ${streamClientBinSrc} )
target_link_libraries( lofarClient ${PROJECT_LIBRARIES} )
#add_executable( lofarServerClient ${serverClientBinSrc} )
#add_executable( lofarServer ${serverBinSrc} )
