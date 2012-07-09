# +-----------------------------------------------------------------------------+
# | $Id:: FindPGPLOT.cmake 3830 2009-12-09 14:35:58Z alexov                   $ |
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

# - Check for the presence of the PGPLOT library
#
# Defines the following variables:
#  HAVE_PGPLOT        = 
#  PGPLOT_INCLUDES    = Path to the PGPLOT header files
#  PGPLOT_LIBRARIES   = Path to all parts of the PGPLOT library
#  PGPLOT_LIBRARY_DIR = Path to the directory containing the PGPLOT libraries

## -----------------------------------------------------------------------------
## Standard locations where to look for required components

#include (CMakeSettings)

## -----------------------------------------------------------------------------
## Check for the header files

FIND_PATH (PGPLOT_INCLUDES cpgplot.h
  PATHS ${include_locations} ${lib_locations}
  PATH_SUFFIXES pgplot
  )

## -----------------------------------------------------------------------------
## Check for the library

set (PGPLOT_LIBRARIES "")

## [1] libpgplot

find_library (PGPLOT_PGPLOT_LIBRARY
  NAMES pgplot
  PATHS ${lib_locations}
  PATH_SUFFIXES pgplot
  NO_DEFAULT_PATH
  )

if (PGPLOT_PGPLOT_LIBRARY)
  list (APPEND PGPLOT_LIBRARIES ${PGPLOT_PGPLOT_LIBRARY})
  get_filename_component(PGPLOT_LIBRARY_DIR ${PGPLOT_PGPLOT_LIBRARY} PATH)
else (PGPLOT_PGPLOT_LIBRARY)
  message (STATUS "Warning: Unable to locate libpgplot!")
endif (PGPLOT_PGPLOT_LIBRARY)

## [2] libcpgplot

find_library (PGPLOT_CPGPLOT_LIBRARY
  NAMES cpgplot
  PATHS ${lib_locations}
  PATH_SUFFIXES pgplot
  NO_DEFAULT_PATH
  )

if (PGPLOT_CPGPLOT_LIBRARY)
  list (APPEND PGPLOT_LIBRARIES ${PGPLOT_CPGPLOT_LIBRARY})
else (PGPLOT_CPGPLOT_LIBRARY)
  message (STATUS "Warning: Unable to locate libcpgplot!")
endif (PGPLOT_CPGPLOT_LIBRARY)

## [3] libXmPgplot

find_library (PGPLOT_XMPGPLOT_LIBRARY
  NAMES XmPgplot
  PATHS ${lib_locations}
  PATH_SUFFIXES pgplot
  NO_DEFAULT_PATH
  )

if (PGPLOT_XMPGPLOT_LIBRARY)
  list (APPEND PGPLOT_LIBRARIES ${PGPLOT_XMPGPLOT_LIBRARY})
else (PGPLOT_XMPGPLOT_LIBRARY)
  message (STATUS "Warning: Unable to locate libXmPgplot!")
endif (PGPLOT_XMPGPLOT_LIBRARY)

## -----------------------------------------------------------------------------
## Actions taken when all components have been found

IF (PGPLOT_INCLUDES AND PGPLOT_LIBRARIES)
  SET (HAVE_PGPLOT TRUE)
ELSE (PGPLOT_INCLUDES AND PGPLOT_LIBRARIES)
  SET (HAVE_PGPLOT FALSE)
  IF (NOT PGPLOT_FIND_QUIETLY)
    IF (NOT PGPLOT_INCLUDES)
      MESSAGE (STATUS "Unable to find PGPLOT header files!")
    ENDIF (NOT PGPLOT_INCLUDES)
    IF (NOT PGPLOT_LIBRARIES)
      MESSAGE (STATUS "Unable to find PGPLOT library files!")
    ENDIF (NOT PGPLOT_LIBRARIES)
  ENDIF (NOT PGPLOT_FIND_QUIETLY)
ENDIF (PGPLOT_INCLUDES AND PGPLOT_LIBRARIES)

if (HAVE_PGPLOT)
  if (NOT PGPLOT_FIND_QUIETLY)
    message (STATUS "Found components for PGPLOT")
    message (STATUS "PGPLOT_INCLUDES  = ${PGPLOT_INCLUDES}")
    message (STATUS "PGPLOT_LIBRARIES = ${PGPLOT_LIBRARIES}")
  endif (NOT PGPLOT_FIND_QUIETLY)
else (HAVE_PGPLOT)
  if (PGPLOT_FIND_REQUIRED)
    message (FATAL_ERROR "Could not find PGPLOT!")
  endif (PGPLOT_FIND_REQUIRED)
endif (HAVE_PGPLOT)

## -----------------------------------------------------------------------------
## Mark as advanced ...

mark_as_advanced (
  PGPLOT_INCLUDES
  PGPLOT_LIBRARIES
  PGPLOT_PGPLOT_LIBRARY
  PGPLOT_CPGPLOT_LIBRARY
  PGPLOT_XMPGPLOT_LIBRARY
  PGPLOT_LIBRARY_DIR
  )

