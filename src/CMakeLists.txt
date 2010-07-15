# === Set the project name and specify the required CMAKE version.
project(pelican-lofar CXX C)
cmake_minimum_required(VERSION 2.4)

# === Project version
set(${PROJECT_NAME}_MAJOR_VERSION 0)
set(${PROJECT_NAME}_MINOR_VERSION 0)
set(${PROJECT_NAME}_PATCH_LEVEL 0)

# === Append the src/cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# === Enable the cmake testing framework.
enable_testing()

# === Options and defaults.
if(NOT PELICAN_LOFAR_INSTALL_DIR)
    set(PELICAN_LOFAR_INSTALL_DIR "pelican-lofar")
endif(NOT PELICAN_LOFAR_INSTALL_DIR)

if(NOT LIBRARY_INSTALL_DIR)
    set(LIBRARY_INSTALL_DIR "${PELICAN_LOFAR_INSTALL_DIR}/lib")
endif(NOT LIBRARY_INSTALL_DIR)

if(NOT INCLUDE_INSTALL_DIR)
    set(INCLUDE_INSTALL_DIR "${PELICAN_LOFAR_INSTALL_DIR}/include")
endif(NOT INCLUDE_INSTALL_DIR)

if(NOT BINARY_INSTALL_DIR)
    set(BINARY_INSTALL_DIR "${PELICAN_LOFAR_INSTALL_DIR}/bin")
endif(NOT BINARY_INSTALL_DIR)

include(compilerDefaults)

# === Find Dependencies.
find_package(Pelican REQUIRED)
find_package(CppUnit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(Boost REQUIRED)
#find_package(MKL QUIET)
#if(MKL_FOUND)
#    add_definitions(-DUSING_MKL)
#    set(CBLAS_LIBS ${MKL_LIBRARIES})
#    include_directories(${MKL_INCLUDE_DIR})
#    message(STATUS "Using MKL CBLAS ${MKL_LIBRARIES}")
#else(MKL_FOUND)
#    find_package(CBLAS REQUIRED)
#    set(CBLAS_LIBS ${CBLAS_LIBRARIES})
#endif(MKL_FOUND)


# === Add directories to the include path (i.e. -I<directory> ...)
include_directories(${PELICAN_INCLUDES})

# == Recurse into sub-directories defined in modules.cmake.
include(modules.cmake)

# === Create the stream client binary.
add_executable(lofarClient
    lofarClient.cpp
    # pipeline.cpp
)
target_link_libraries(lofarClient
    pelican-lofar_static
    ${PELICAN_LIBRARY}
    ${FFTW3_LIBRARIES}
)

# === Create the server binary.
#set( serverBinSrc
#     src/serverBin.cpp
#)

#add_executable(lofarServerClient ${serverClientBinSrc})
#add_executable(lofarServer ${serverBinSrc})
