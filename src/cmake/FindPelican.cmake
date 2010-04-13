# - Find pelican
# Find the native PELICAN includes and library
#
#  PELICAN_FOUND         = True if cfitsio found
#  PELICAN_LIBRARIES     = Set of libraries required for linking
#  PELICAN_INCLUDE_DIR   = Directory where to find fitsio.h
#  PELICAN_LIBRARY_pelican  = the pelican library file

# Already in cache, be silent
#IF (PELICAN_INCLUDE_DIR)
#    SET(PELICAN_FIND_QUIETLY TRUE)
#ENDIF (PELICAN_INCLUDE_DIR)


# QT4 Core and XML components are required by pelican.
find_package(Qt4 COMPONENTS QtCore QtXml QtNetwork REQUIRED)

FIND_PATH(PELICAN_INCLUDE_DIR pelican PATHS /usr/include/ /usr/local/include )

## =============================================================================
## =============================================================================
list(APPEND PELICAN_INCLUDE_DIR ${PELICAN_INCLUDE_DIR}/pelican) # TODO REMOVE
## =============================================================================
## =============================================================================

SET(PELICAN_NAMES pelican)

FOREACH(lib ${PELICAN_NAMES} )
    FIND_LIBRARY(PELICAN_LIBRARY_${lib} NAMES ${lib})
    LIST(APPEND PELICAN_LIBRARIES ${PELICAN_LIBRARY_${lib}})
ENDFOREACH(lib)

# handle the QUIETLY and REQUIRED arguments and set PELICAN_FOUND to TRUE if.
# all listed variables are TRUE
include(FindPackageHandleCompat)
#include(FindPackageHandleStandardArgs) ??! maybe ??!
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Pelican DEFAULT_MSG PELICAN_LIBRARIES PELICAN_INCLUDE_DIR)

# Append Qt stuff (pelican depends on these)
list(APPEND PELICAN_LIBRARIES
    ${QT_QTCORE_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
)
list(APPEND PELICAN_INCLUDE_DIR
    ${QT_INCLUDE_DIR}
    ${QT_QTCORE_INCLUDE_DIR}
    ${QT_QTXML_INCLUDE_DIR}
    ${QT_QTNETWORK_INCLUDE_DIR}
)

if (NOT PELICAN_FOUND)
    set(PELICAN_LIBRARIES)
endif (NOT PELICAN_FOUND)

MARK_AS_ADVANCED(PELICAN_LIBRARIES PELICAN_INCLUDE_DIR)
