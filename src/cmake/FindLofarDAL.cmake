# - Find cppunit
# Find the native LOFAR_DAL includes and library
#
#  LOFAR_DAL_INCLUDE_DIR - where to find dal.h.
#  LOFAR_DAL_LIBRARY_DIR - where to find dal.h.
#  LOFAR_DAL_INSTALL_DIR - top where lofar dal has been installed (lib and include dirs included)
#  LOFAR_DAL_LIBRARIES   - List of libraries when using cppunit.
#  LOFAR_DAL_FOUND       - True if cppunit found.

# Already in cache, be silent
IF (LOFAR_DAL_INCLUDE_DIR)
    SET(LOFAR_DAL_FIND_QUIETLY TRUE)
ENDIF (LOFAR_DAL_INCLUDE_DIR)

FIND_PATH(LOFAR_DAL_INCLUDE_DIR dal/dal_config.h 
    PATHS ${LOFAR_DAL_INCLUDE_DIR}
	  ${LOFAR_DAL_INSTALL_DIR}/include
	  /usr/local/include 
	  /usr/include )

SET(LOFAR_DAL_NAMES lofardal)
FOREACH( lib ${LOFAR_DAL_NAMES} )
    FIND_LIBRARY(LOFAR_DAL_LIBRARY_${lib} 
	NAMES ${lib}
	PATHS ${LOFAR_DAL_LIBRARY_DIR} ${LOFAR_DAL_INSTALL_DIR}/lib /usr/local/lib /usr/lib
    )
    LIST(APPEND LOFAR_DAL_LIBRARIES ${LOFAR_DAL_LIBRARY_${lib}})
ENDFOREACH(lib)

# handle the QUIETLY and REQUIRED arguments and set LOFAR_DAL_FOUND to TRUE if.
# all listed variables are TRUE
INCLUDE(FindPackageHandleCompat)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LOFAR_DAL DEFAULT_MSG LOFAR_DAL_LIBRARIES LOFAR_DAL_INCLUDE_DIR)

IF(NOT LOFAR_DAL_FOUND)
    SET( LOFAR_DAL_LIBRARIES )
ENDIF(NOT LOFAR_DAL_FOUND)

MARK_AS_ADVANCED(LOFAR_DAL_LIBRARIES LOFAR_DAL_INCLUDE_DIR)

