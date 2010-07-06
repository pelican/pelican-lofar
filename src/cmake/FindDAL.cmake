# +-----------------------------------------------------------------------------+
# | $Id:: FindDAL.cmake 4887 2010-05-13 17:58:33Z baehren                     $ |
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

# Check for the presence of Data Access Library (DAL).
#
# The following variables are set when DAL is found:
#  HAVE_DAL       = Set to true, if all components of DAL have been found.
#  DAL_INCLUDES   = Include path for the header files of DAL
#  DAL_LIBRARIES  = Link these to use DAL
#  DAL_LFGLAS     = Linker flags (optional)

if (NOT FIND_DAL_CMAKE)
  
  set (FIND_DAL_CMAKE TRUE)
  
  ##_____________________________________________________________________________
  ## Search locations
  
  include (CMakeSettings)
  
  ##_____________________________________________________________________________
  ## Check for the header files
  
  find_path (DAL_INCLUDES dalCommon.h dalData.h dalFilter.h
    PATHS ${include_locations}
    PATH_SUFFIXES dal
    NO_DEFAULT_PATH
    )
  
  get_filename_component (DAL_INCLUDES ${DAL_INCLUDES} ABSOLUTE)
  
  ##_____________________________________________________________________________
  ## Check for the library
  
  find_library (DAL_LIBRARIES dal
    PATHS ${lib_locations}
    NO_DEFAULT_PATH
    )
  
  ##_____________________________________________________________________________
  ## Actions taken when all components have been found
  
  if (DAL_INCLUDES AND DAL_LIBRARIES)
    set (HAVE_DAL TRUE)
  else (DAL_INCLUDES AND DAL_LIBRARIES)
    set (HAVE_DAL FALSE)
    if (NOT DAL_FIND_QUIETLY)
      if (NOT DAL_INCLUDES)
	message (STATUS "Unable to find DAL header files!")
      endif (NOT DAL_INCLUDES)
      if (NOT DAL_LIBRARIES)
	message (STATUS "Unable to find DAL library files!")
      endif (NOT DAL_LIBRARIES)
    endif (NOT DAL_FIND_QUIETLY)
  endif (DAL_INCLUDES AND DAL_LIBRARIES)
  
  if (HAVE_DAL)
    if (NOT DAL_FIND_QUIETLY)
      message (STATUS "Found components for DAL")
      message (STATUS "DAL_INCLUDES  = ${DAL_INCLUDES}")
      message (STATUS "DAL_LIBRARIES = ${DAL_LIBRARIES}")
    endif (NOT DAL_FIND_QUIETLY)
  else (HAVE_DAL)
    if (DAL_FIND_REQUIRED)
      message (FATAL_ERROR "Could not find DAL!")
    endif (DAL_FIND_REQUIRED)
  endif (HAVE_DAL)
  
  ##_____________________________________________________________________________
  ## Mark advanced variables
  
  mark_as_advanced (
    DAL_INCLUDES
    DAL_LIBRARIES
    )
  
endif (NOT FIND_DAL_CMAKE)
