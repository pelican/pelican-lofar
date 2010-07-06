# +-----------------------------------------------------------------------------+
# | $Id:: FindHDF5.cmake 4888 2010-05-13 17:58:34Z baehren                    $ |
# +-----------------------------------------------------------------------------+
# |   Copyright (C) 2007                                                        |
# |   Lars B"ahren (bahren@astron.nl)                                           |
# |                                                                             |
# |   This program is free software; you can redistribute it and/or modify      |
# |   it under the terms of the GNU General Public License as published by      |
# |   the Free Software Foundation; either version 2 of the License, or         |
# |   (at your option) any later version.                                       |
# |                                                                             |
# |   This program is distributed in the hope that it will be useful,           |
# |   but WITHOUT ANY WARRANTY; without even the implied warranty of            |
# |   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             |
# |   GNU General Public License for more details.                              |
# |                                                                             |
# |   You should have received a copy of the GNU General Public License         |
# |   along with this program; if not, write to the                             |
# |   Free Software Foundation, Inc.,                                           |
# |   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.                 |
# +-----------------------------------------------------------------------------+

# - Check for the presence of HDF5
#
# The following variables are set when HDF5 is found:
#  HAVE_HDF5            = Set to true, if all components of HDF5 have been found.
#  HDF5_INCLUDES        = Include path for the header files of HDF5
#  HDF5_HDF5_LIBRARY    = Path to libhdf5
#  HDF5_HDF5_HL_LIBRARY = Path to libhdf5_hl, the high-level interface
#  HDF5_LIBRARIES       = Link these to use HDF5
#  HDF5_MAJOR_VERSION   = Major version of the HDF5 library
#  HDF5_MINOR_VERSION   = Minor version of the HDF5 library
#  HDF5_RELEASE_VERSION = Release version of the HDF5 library

if (NOT FIND_HDF5_CMAKE)

  set (FIND_HDF5_CMAKE TRUE)
  
  ##_____________________________________________________________________________
  ## Search locations
  
  include (CMakeSettings)
  
  ##_____________________________________________________________________________
  ## Check for the header files
  
  find_path (HDF5_INCLUDES hdf5.h hdf5_hl.h
    PATHS ${include_locations}
    PATH_SUFFIXES hdf5
    NO_DEFAULT_PATH
    )
  
  ## search for individual header files
  
  find_path (HAVE_HDF5_HDF5_H hdf5.h
    PATHS ${include_locations}
    PATH_SUFFIXES hdf5
    NO_DEFAULT_PATH
    )
  
  find_path (HAVE_HDF5_H5LT_H H5LT.h
    PATHS ${include_locations}
    PATH_SUFFIXES hdf5
    NO_DEFAULT_PATH
    )
  
  find_path (HAVE_HDF5_HDF5_HL_H hdf5_hl.h
    PATHS ${include_locations}
    PATH_SUFFIXES hdf5
    NO_DEFAULT_PATH
    )
  
  ##_____________________________________________________________________________
  ## Check for the library components
  
  ## [1] Core library (libhdf5)
  
  find_library (HDF5_HDF5_LIBRARY
    NAMES hdf5
    PATHS ${lib_locations}
    PATH_SUFFIXES hdf5
    NO_DEFAULT_PATH
    )
  
  if (HDF5_HDF5_LIBRARY)
    set (HDF5_LIBRARIES ${HDF5_HDF5_LIBRARY})
  endif (HDF5_HDF5_LIBRARY)
  
  ## [2] High level interface (libhdf5_hl)
  
  FIND_LIBRARY (HDF5_HDF5_HL_LIBRARY
    NAMES hdf5_hl
    PATHS ${lib_locations}
    PATH_SUFFIXES hdf5
    NO_DEFAULT_PATH
    )
  
  if (HDF5_HDF5_HL_LIBRARY)
    list (APPEND HDF5_LIBRARIES ${HDF5_HDF5_HL_LIBRARY})
  endif (HDF5_HDF5_HL_LIBRARY)
  
  ## [3] C++ interface (libhdf5_cpp)
  
  FIND_LIBRARY (HDF5_HDF5_CPP_LIBRARY
    NAMES hdf5_cpp
    PATHS ${lib_locations}
    PATH_SUFFIXES hdf5
    NO_DEFAULT_PATH
    )
  
  if (HDF5_HDF5_CPP_LIBRARY)
    list (APPEND HDF5_LIBRARIES ${HDF5_HDF5_CPP_LIBRARY})
  endif (HDF5_HDF5_CPP_LIBRARY)
  
  ##_____________________________________________________________________________
  ## Actions taken when all components have been found
  
  if (HDF5_INCLUDES AND HDF5_LIBRARIES)
    set (HAVE_HDF5 TRUE)
  else (HDF5_INCLUDES AND HDF5_LIBRARIES)
    set (HAVE_HDF5 FALSE)
    if (NOT HDF5_FIND_QUIETLY)
      if (NOT HDF5_INCLUDES)
	message (STATUS "Unable to find HDF5 header files!")
      endif (NOT HDF5_INCLUDES)
      if (NOT HDF5_LIBRARIES)
	message (STATUS "Unable to find HDF5 library files!")
      endif (NOT HDF5_LIBRARIES)
    endif (NOT HDF5_FIND_QUIETLY)
  endif (HDF5_INCLUDES AND HDF5_LIBRARIES)
  
  ##_____________________________________________________________________________
  ## Determine library version
  
  find_file (HAVE_H5PUBLIC_H H5public.h
    PATHS ${include_locations}
    PATH_SUFFIXES hdf5
    NO_DEFAULT_PATH
    )
  
  if (HAVE_H5PUBLIC_H)
    
    ## extract library major version
    file (STRINGS ${HAVE_H5PUBLIC_H} HDF5_MAJOR_VERSION
      REGEX "H5_VERS_MAJOR.*For major"
      )
    string (REGEX REPLACE "#define H5_VERS_MAJOR" "" HDF5_MAJOR_VERSION ${HDF5_MAJOR_VERSION})
    string (REGEX MATCH "[0-9]" HDF5_MAJOR_VERSION ${HDF5_MAJOR_VERSION})
    ## extract library minor version
    file (STRINGS ${HAVE_H5PUBLIC_H} HDF5_MINOR_VERSION
      REGEX "H5_VERS_MINOR.*For minor"
      )
    string (REGEX REPLACE "#define H5_VERS_MINOR" "" HDF5_MINOR_VERSION ${HDF5_MINOR_VERSION})
    string (REGEX MATCH "[0-9]" HDF5_MINOR_VERSION ${HDF5_MINOR_VERSION})
    ## extract library release version
    file (STRINGS ${HAVE_H5PUBLIC_H} HDF5_RELEASE_VERSION
      REGEX "H5_VERS_RELEASE.*.For tweaks"
      )
    string (REGEX REPLACE "#define H5_VERS_RELEASE" "" HDF5_RELEASE_VERSION ${HDF5_RELEASE_VERSION})
    string (REGEX MATCH "[0-9]" HDF5_RELEASE_VERSION ${HDF5_RELEASE_VERSION})
  else (HAVE_H5PUBLIC_H)
    find_file (HAVE_TESTHDF5VERSION TestHDF5Version.cc
      PATHS ${CMAKE_MODULE_PATH} ${USG_ROOT}
      PATH_SUFFIXES devel_common/cmake
      )
    
  else (HAVE_H5PUBLIC_H)
    
    if (HAVE_HDF5 AND HAVE_TESTHDF5VERSION)
      
      list (APPEND CMAKE_REQUIRED_LIBRARIES ${HDF5_LIBRARIES})
      
      try_run (var_exit var_compiled
	${CMAKE_BINARY_DIR}
	${HAVE_TESTHDF5VERSION}
	COMPILE_DEFINITIONS -I${HDF5_INCLUDES} -I${HDF5_INCLUDES}/hdf5
	CMAKE_FLAGS -DHAVE_H5PUBLIC_H:BOOL=${HAVE_H5PUBLIC_H} -DLINK_LIBRARIES:STRING=${CMAKE_REQUIRED_LIBRARIES}
	COMPILE_OUTPUT_VARIABLE var_compile
	RUN_OUTPUT_VARIABLE var_run
	)
      
      ## process the output of the test program
      
      if (var_compiled AND var_exit)
	
	string (REGEX MATCH "maj=.*:min" HDF5_MAJOR_VERSION ${var_run})
	string (REGEX REPLACE "maj=" "" HDF5_MAJOR_VERSION ${HDF5_MAJOR_VERSION})
	string (REGEX REPLACE ":min" "" HDF5_MAJOR_VERSION ${HDF5_MAJOR_VERSION})
	
	string (REGEX MATCH "min=.*:rel" HDF5_MINOR_VERSION ${var_run})
	string (REGEX REPLACE "min=" "" HDF5_MINOR_VERSION ${HDF5_MINOR_VERSION})
	string (REGEX REPLACE ":rel" "" HDF5_MINOR_VERSION ${HDF5_MINOR_VERSION})
	
	string (REGEX MATCH "rel=.*:" HDF5_RELEASE_VERSION ${var_run})
	string (REGEX REPLACE "rel=" "" HDF5_RELEASE_VERSION ${HDF5_RELEASE_VERSION})
	string (REGEX REPLACE ":" "" HDF5_RELEASE_VERSION ${HDF5_RELEASE_VERSION})
	
      else (var_compiled AND var_exit)
	message (STATUS "[FindHDF5] Unable to determine HDF5 library version!")
	## did we at least manage to compile the source?
	if (NOT var_compiled)
	  message (STATUS "Compile of test program failed! -- ${var_compile}")
	endif (NOT var_compiled)
      endif (var_compiled AND var_exit)
      
    endif (HAVE_HDF5 AND HAVE_TESTHDF5VERSION)
    
  endif (HAVE_H5PUBLIC_H)
  
  ##_____________________________________________________________________________
  ## Feedback
  
  if (HAVE_HDF5)
    if (NOT HDF5_FIND_QUIETLY)
      message (STATUS "Found components for HDF5")
      message (STATUS "HDF5_INCLUDES        = ${HDF5_INCLUDES}")
      message (STATUS "HDF5_LIBRARIES       = ${HDF5_LIBRARIES}")
      message (STATUS "HDF5_MAJOR_VERSION   = ${HDF5_MAJOR_VERSION}")
      message (STATUS "HDF5_MINOR_VERSION   = ${HDF5_MINOR_VERSION}")
      message (STATUS "HDF5_RELEASE_VERSION = ${HDF5_RELEASE_VERSION}")
    endif (NOT HDF5_FIND_QUIETLY)
  else (HAVE_HDF5)
    if (HDF5_FIND_REQUIRED)
      message (FATAL_ERROR "Could not find HDF5!")
    endif (HDF5_FIND_REQUIRED)
  endif (HAVE_HDF5)
  
  ##_____________________________________________________________________________
  ## Mark as advanced ...
  
  mark_as_advanced (
    HDF5_INCLUDES
    HDF5_LIBRARIES
    HAVE_H5PUBLIC_H
    HDF5_MAJOR_VERSION
    HDF5_MINOR_VERSION
    HDF5_RELEASE_VERSION
    HAVE_TESTHDF5VERSION
    HDF5_HDF5_LIBRARY
    HDF5_HDF5_HL_LIBRARY
    HDF5_HDF5_CPP_LIBRARY
    )
  
endif (NOT FIND_HDF5_CMAKE)
