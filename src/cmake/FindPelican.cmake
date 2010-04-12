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

FIND_PATH(PELICAN_INCLUDE_DIR pelican PATHS /usr/include/ /usr/local/include )

SET(PELICAN_NAMES pelican)
    
FOREACH(lib ${PELICAN_NAMES} )
    FIND_LIBRARY(PELICAN_LIBRARY_${lib} NAMES ${lib})
    LIST(APPEND PELICAN_LIBRARIES ${PELICAN_LIBRARY_${lib}})
ENDFOREACH(lib)

# handle the QUIETLY and REQUIRED arguments and set PELICAN_FOUND to TRUE if.
# all listed variables are TRUE
INCLUDE(FindPackageHandleCompat)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PELICAN DEFAULT_MSG PELICAN_LIBRARIES PELICAN_INCLUDE_DIR)

IF(NOT PELICAN_FOUND)
    SET( PELICAN_LIBRARIES )
ENDIF(NOT PELICAN_FOUND)

MARK_AS_ADVANCED(PELICAN_LIBRARIES PELICAN_INCLUDE_DIR)
