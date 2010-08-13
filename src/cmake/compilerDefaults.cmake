#
# Compiler defaults for pelican.
# This file is included in the top level pelican CMakeLists.txt.
#

#=== Build in debug mode if not otherwise specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT CMAKE_BUILD_TYPE MATCHES "^RELEASE|DEBUG|[Rr]elease|[Dd]ebug$")
    message(FATAL_ERROR "## Unknown build type. Select 'debug' or 'release'")
endif(NOT CMAKE_BUILD_TYPE MATCHES "^RELEASE|DEBUG|[Rr]elease|[Dd]ebug$")

# Set the C++ release flags.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DNDEBUG")

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -Wextra)
    add_definitions(-Wno-deprecated -Wno-unknown-pragmas)
    list(APPEND CPP_PLATFORM_LIBS util dl)
elseif(CMAKE_CXX_COMPILER MATCHES icpc)
    add_definitions(-Wall -Wcheck)
    add_definitions(-wd383 -wd981 -wd444)  # Suppress remarks / warnings.
    add_definitions(-ww111 -ww1572) # Promote remarks to warnings.
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DNDEBUG -xSSSE3") # Core 2
    #set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DNDEBUG -xSSE4.2") # i7
    #set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DNDEBUG")
else(CMAKE_COMPILER_IS_GNUCXX)
    # use defaults (and pray it works...)
endif(CMAKE_COMPILER_IS_GNUCXX)

if(APPLE)
    add_definitions(-DDARWIN)
endif(APPLE)




