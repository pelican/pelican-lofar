
## -----------------------------------------------------------------------------
## $Id:: CMakeSettings.cmake 5035 2010-06-03 15:18:56Z baehren                 $
## -----------------------------------------------------------------------------

## Variables used through the configuration environment:
##
##  USG_ROOT              -- Root of the USG directory tree.
##  USG_CMAKE_CONFIG      -- Cache variable used to control running through the
##                           common set of instructions provided with this file.
##                           since this file will be included multiple times
##                           during the configuration process to a project, this
##                           variable serves as an include-guard, both protecting
##                           previously assigned variables as well as avoiding 
##                           unnecessary passes through the instructions.
##  USG_LIB_LOCATIONS     -- 
##  USG_INCLUDE_LOCATIONS -- 
##  USG_INSTALL_PREFIX    -- Prefix marking the location at which the finished
##                           software components will be installed
##  USG_VARIANTS_FILE     -- Variants file containing host-specific overrides
##                           to the common configuration settings/presets.
##

if (NOT USG_CMAKE_CONFIG)

  ##________________________________________________________
  ## Check if USG_ROOT is defined

  if (NOT USG_ROOT)
    message (STATUS "[USG CMake] USG_ROOT undefined; trying to locate it...")
    ## try to find the root directory based on the location of the release
    ## directory
    find_path (USG_INSTALL_PREFIX release/release_area.txt
      $ENV{LOFARSOFT}
      ${CMAKE_CURRENT_SOURCE_DIR}/..
      ${CMAKE_CURRENT_SOURCE_DIR}/../..
      ${CMAKE_CURRENT_SOURCE_DIR}/../../..
      NO_DEFAULT_PATH
      )
    ## convert the relative path to an absolute one
    get_filename_component (USG_ROOT ${USG_INSTALL_PREFIX} ABSOLUTE)
  endif (NOT USG_ROOT)

  ## Second pass: check once more if USG_ROOT is defined
  
  if (USG_ROOT)
    ## This addition to the module path needs to go into the cache,
    ## because otherwise it will be gone at the next time CMake is run
    set (CMAKE_MODULE_PATH ${USG_ROOT}/devel_common/cmake CACHE PATH
      "USG cmake modules"
      FORCE)
    ## installation location
    set (USG_INSTALL_PREFIX ${USG_ROOT}/release CACHE PATH
      "USG default install area"
      FORCE
      )
    set (CMAKE_INSTALL_PREFIX ${USG_ROOT}/release CACHE PATH
      "CMake installation area" 
      FORCE
      )
    ## header files
    include_directories (${USG_ROOT}/release/include CACHE PATH
      "USG include area"
      FORCE
      )
    ## (Test) data
    set (USG_DATA ${USG_ROOT}/data CACHE PATH
      "USG data area"
      FORCE
      )
    ## USG augmentation to PYTHONPATH
    set (USG_PYTHONPATH ${USG_ROOT}/release/lib/python2.6;${USG_ROOT}/release/lib/python2.5
      CACHE
      PATH
      "USG data area"
      FORCE
      )
    ## Directories inside the release directory
    if (USG_INSTALL_PREFIX)
      execute_process (
	COMMAND mkdir -p lib
	COMMAND ln -s lib lib64
	WORKING_DIRECTORY ${USG_INSTALL_PREFIX}
	)
    endif (USG_INSTALL_PREFIX)
  else (USG_ROOT)
    message (SEND_ERROR "USG_ROOT is undefined!")
  endif (USG_ROOT)
  
  ## ---------------------------------------------------------------------------
  ## generic search locations

  set (search_locations
    ${USG_INSTALL_PREFIX}
    /opt
    /opt/local
    /sw
    /usr
    /usr/local
    /usr/X11R6
    /opt/casa/local
    /app/usg
    CACHE
    PATH
    "Directories to look for include files"
    FORCE
    )

  ## ---------------------------------------------------------------------------
  ## locations in which to look for applications/binaries
  
  set (bin_locations
    ${USG_INSTALL_PREFIX}/bin
    /opt/local/bin
    /sw/bin
    /usr/bin
    /usr/local/bin
    /app/usg/release/bin
    CACHE
    PATH
    "Extra directories to look for executable files"
    FORCE
    )
  
  ## ----------------------------------------------------------------------------
  ## locations in which to look for header files
  
  set (include_locations
    ${USG_INSTALL_PREFIX}/include
    /opt/include
    /opt/local/include
    /sw/include
    /usr/include
    /usr/local/include
    /usr/X11R6/include
    /opt/casa/local/include    
    /app/usg/release/include
    /Developer/SDKs/MacOSX10.5.sdk/usr/include
    CACHE
    PATH
    "Directories to look for include files"
    FORCE
    )
  
  ## ----------------------------------------------------------------------------
  ## locations in which to look for libraries
  
  set (lib_locations
    ${USG_INSTALL_PREFIX}/lib
    ${USG_INSTALL_PREFIX}/lib64
    /opt/lib
    /opt/local/lib
    /sw/lib
    /usr/local/lib64
    /usr/local/lib
    /usr/lib64
    /usr/lib
    /usr/X11R6/lib
    /Developer/SDKs/MacOSX10.4u.sdk/usr/lib
    /Developer/SDKs/MacOSX10.5.sdk/usr/lib
    /app/usg/release/lib
    CACHE
    PATH
    "Directories to look for libraries"
    FORCE
    )

  ## ============================================================================
  ##
  ##  Check for test datasets
  ##
  ## ============================================================================

  include (FindTestDatasets)

  ## ============================================================================
  ##
  ##  System inspection
  ##
  ## ============================================================================

  ##________________________________________________________
  ## Size of variable types
  
  include (CheckTypeSize)
  
  check_type_size ("short"          SIZEOF_SHORT         )
  check_type_size ("int"            SIZEOF_INT           )
  check_type_size ("float"          SIZEOF_FLOAT         )
  check_type_size ("double"         SIZEOF_DOUBLE        )
  check_type_size ("long"           SIZEOF_LONG          )
  check_type_size ("long int"        SIZEOF_LONG_INT     )
  check_type_size ("long long"      SIZEOF_LONG_LONG     )
  check_type_size ("long long int"  SIZEOF_LONG_LONG_INT )
  check_type_size ("uint"           SIZEOF_UINT          )
  check_type_size ("int8_t"         SIZEOF_INT8_T        )
  check_type_size ("int16_t"        SIZEOF_INT16_T       )
  check_type_size ("int32_t"        SIZEOF_INT32_T       )
  check_type_size ("int64_t"        SIZEOF_INT64_T       )
  check_type_size ("uint8_t"        SIZEOF_UINT8_T       )
  check_type_size ("uint16_t"       SIZEOF_UINT16_T      )
  check_type_size ("uint32_t"       SIZEOF_UINT32_T      )
  check_type_size ("uint64_t"       SIZEOF_UINT64_T      )
  
  if (CMAKE_SIZEOF_VOID_P)
    if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
      add_definitions (-DWORDSIZE_IS_64)
    endif (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
  else (CMAKE_SIZEOF_VOID_P)
    message (STATUS "Unable to check size of void*")
  endif (CMAKE_SIZEOF_VOID_P)
  
  if (UNIX)
    execute_process (
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      COMMAND uname -m
      TIMEOUT 20
      OUTPUT_VARIABLE CMAKE_SYSTEM_KERNEL
      )
  endif (UNIX)
  
  ##__________________________________________________________
  ## System header files
  
  find_path (HAVE_LIBGEN_H      libgen.h      PATHS ${include_locations} )
  find_path (HAVE_LIMITS_H      limits.h      PATHS ${include_locations} )
  find_path (HAVE_MATH_H        math.h        PATHS ${include_locations} )
  find_path (HAVE_MEMORY_H      memory.h      PATHS ${include_locations} )
  find_path (HAVE_STDINT_H      stdint.h      PATHS ${include_locations} )
  find_path (HAVE_STDIO_H       stdio.h       PATHS ${include_locations} )
  find_path (HAVE_STDLIB_H      stdlib.h      PATHS ${include_locations} )
  find_path (HAVE_STRING_H      string.h      PATHS ${include_locations} )
  find_path (HAVE_STRINGS_H     strings.h     PATHS ${include_locations} )
  find_path (HAVE_TIME_H        time.h        PATHS ${include_locations} )
  find_path (HAVE_SYS_SOCKET_H  sys/socket.h  PATHS ${include_locations} )
  find_path (HAVE_SYS_STAT_H    sys/stat.h    PATHS ${include_locations} )
  find_path (HAVE_SYS_SYSCTL_H  sys/sysctl.h  PATHS ${include_locations} )
  find_path (HAVE_SYS_TIME_H    sys/time.h    PATHS ${include_locations} )
  find_path (HAVE_SYS_TYPES_H   sys/types.h   PATHS ${include_locations} )
  find_path (HAVE_SYS_UTIME_H   sys/utime.h   PATHS ${include_locations} )

  ##__________________________________________________________
  ## System Libraries
  
  find_library (HAVE_LIBM        m        PATHS ${lib_locations} )
  find_library (HAVE_LIBUTIL     util     PATHS ${lib_locations} )
  find_library (HAVE_LIBDL       dl       PATHS ${lib_locations} )
  find_library (HAVE_LIBGD       gd       PATHS ${lib_locations} )
  find_library (HAVE_LIBPTHREAD  pthread  PATHS ${lib_locations} )
  find_library (HAVE_LIBZ        z        PATHS ${lib_locations} )
  
  ## ============================================================================
  ##
  ##  Internal CMake variables
  ##
  ## ============================================================================
  
  if (APPLE)
    if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES powerpc)
#      set (CMAKE_OSX_ARCHITECTURES i386;x86_64)
      if (CMAKE_SIZEOF_VOID_P)
	if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
	  set (CMAKE_SYSTEM_WORDSIZE 64)
	else (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
	  set (CMAKE_SYSTEM_WORDSIZE 32)
	endif (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
      endif (CMAKE_SIZEOF_VOID_P)
    endif (NOT CMAKE_SYSTEM_PROCESSOR MATCHES powerpc)
  endif (APPLE)
  
  if (UNIX)
    set (CMAKE_FIND_LIBRARY_PREFIXES "lib" CACHE STRING
      "Library prefixes"
      FORCE
      )
    if (NOT APPLE AND NOT CMAKE_FIND_LIBRARY_SUFFIXES)
      set (CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so" CACHE STRING
	"Library suffices"
	FORCE
	)
    endif (NOT APPLE AND NOT CMAKE_FIND_LIBRARY_SUFFIXES)
  endif (UNIX)
  
  set (USG_DOWNLOAD "http://usg.lofar.org/download" CACHE
    STRING
    "URL for the download area on the USG server"
    FORCE
    )
  
  ## ============================================================================
  ##
  ##  Host-specific overrides
  ##
  ## ============================================================================

  execute_process (COMMAND hostname -s
    OUTPUT_VARIABLE hostname
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set (USG_VARIANTS_FILE ${USG_ROOT}/devel_common/cmake/variants.${hostname})
  
  if (EXISTS ${USG_VARIANTS_FILE})
    message (STATUS "Loading settings variants " ${USG_VARIANTS_FILE})
    include (${USG_VARIANTS_FILE})
  endif (EXISTS ${USG_VARIANTS_FILE})
  
  ## ----------------------------------------------------------------------------
  ## Configuration flag
  
  set (USG_CMAKE_CONFIG TRUE CACHE BOOL "USG CMake configuration flag" FORCE)
  mark_as_advanced(USG_CMAKE_CONFIG)
  
  ## ============================================================================
  ##
  ##  Configuration summary
  ##
  ## ============================================================================

  message (STATUS)
  message (STATUS "+------------------------------------------------------------+")
  message (STATUS)

  message (STATUS "[USG CMake configuration]")
  message (STATUS " CMAKE_SYSTEM .............. : ${CMAKE_SYSTEM}"                )
  message (STATUS " CMAKE_SYSTEM_VERSION ...... : ${CMAKE_SYSTEM_VERSION}"        )
  message (STATUS " CMAKE_SYSTEM_PROCESSOR .... : ${CMAKE_SYSTEM_PROCESSOR}"      )
  message (STATUS " CMAKE_SYSTEM_KERNEL ....... : ${CMAKE_SYSTEM_KERNEL}"         )
  message (STATUS " USG_ROOT .................. : ${USG_ROOT}"                    )
  message (STATUS " CMAKE_INSTALL_PREFIX ...... : ${CMAKE_INSTALL_PREFIX}"        )
  message (STATUS " CMAKE_FIND_LIBRARY_PREFIXES : ${CMAKE_FIND_LIBRARY_PREFIXES}" )
  message (STATUS " CMAKE_FIND_LIBRARY_SUFFIXES : ${CMAKE_FIND_LIBRARY_SUFFIXES}" )
  message (STATUS " Size of void* ............. : ${CMAKE_SIZEOF_VOID_P}"         )
  
  message (STATUS)
  message (STATUS "+------------------------------------------------------------+")
  message (STATUS)
  
endif (NOT USG_CMAKE_CONFIG)
