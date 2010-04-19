# FindPelicanInstall.cmake
#
# Finds installed PELICAN includes, library and associated dependencies.
#
# Defines the following variables:
#   PELICAN_FOUND            = True if PELICAN found
#   PELICAN_INCLUDE_DIR      = Top level pelican include directory.
#   PELICAN_INCLUDES         = Set of include directories needed by PELICAN.
#   PELICAN_LIBRARY          = The PELICAN library
#   PELICAN_LIBRARIES        = Set of libraries required for linking.
#

# Find the pelican cmake modules directory.
find_path(PELICAN_CMAKE_MODULE_DIR FindPelicanInstall.cmake
    PATHS
    /usr/
    /usr/share
    /usr/share/pelican
    /usr/share/pelican/cmake
    /usr/local/
    /usr/local/share
    /usr/local/share/pelican
    /usr/local/share/pelican/cmake
    PATH_SUFFIXES
    share
    cmake
    pelican
    DOC
    "Location of Pelican cmake modules."
)

message(STATUS "========== ${PELICAN_CMAKE_MODULE_DIR}")

# Handle the QUIETLY and REQUIRED arguments.
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PelicanInstall
    "ERROR: Could not find Required Pelican cmake module path"
    PELICAN_CMAKE_MODULE_DIR
)

# Add the module directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 "${PELICAN_CMAKE_MODULE_DIR}")
list(INSERT CMAKE_INCLUDE_PATH 0 "${PELICAN_CMAKE_MODULE_DIR}")

# Find the pelican library setting the pelican libraries and includes.
find_package(PelicanInstall REQUIRED)

# handle the QUIETLY and REQUIRED arguments.
# ==============================================================================
include(FindPackageHandleCompat)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PelicanInstall
        DEFAULT_MSG PELICAN_LIBRARIES PELICAN_INCLUDE_DIR)

# Hide in the cache.
# ==============================================================================
#mark_as_advanced()



