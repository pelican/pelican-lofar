#
# Compiler defaults for pelican.
# This file is included in the top level pelican CMakeLists.txt.
#

#=== Build in debug mode if not otherwise specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT CMAKE_BUILD_TYPE MATCHES "^RELEASE|DEBUG|[Rr]elease|[Dd]ebug|[Pp]rofile|PROFILE$")
    message(FATAL_ERROR "## Unknown build type. Select 'debug','release' or 'profile'")
endif(NOT CMAKE_BUILD_TYPE MATCHES "^RELEASE|DEBUG|[Rr]elease|[Dd]ebug|[Pp]rofile|PROFILE$")

message("*****************************************************************")
if (CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
    message("** NOTE: Building in release mode!")
else ()
    message("** NOTE: Building in debug mode!")
endif()
message("*****************************************************************")


#=== Set compiler flags.
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DNDEBUG")
    set(CMAKE_CXX_FLAGS_PROFILE "-pg")
    set(CMAKE_EXE_LINKER_FLAG_PROFILE "-pg")
    add_definitions(-Wall -Wextra)
    add_definitions(-Wno-deprecated -Wno-unknown-pragmas)
    list(APPEND CPP_PLATFORM_LIBS util dl)
elseif(CMAKE_CXX_COMPILER MATCHES icpc)
    add_definitions(-Wall -Wcheck)
    add_definitions(-wd383 -wd981 -wd444)  # Suppress remarks / warnings.
    add_definitions(-ww111 -ww1572) # Promote remarks to warnings.
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DNDEBUG")
else(CMAKE_COMPILER_IS_GNUCXX)
    # use defaults (and pray it works...)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DNDEBUG")
endif(CMAKE_COMPILER_IS_GNUCXX)

if(APPLE)
    add_definitions(-DDARWIN)
endif(APPLE)




