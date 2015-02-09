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
    set(CMAKE_CXX_FLAGS_PROFILE "-pg -DNDEBUG -DQT_NO_DEBUG -O2")
    set(CMAKE_EXE_LINKER_FLAG_PROFILE "-pg")
    add_definitions(-Wall -Wextra -pedantic -Wno-long-long -Wno-variadic-macros)
    add_definitions(-Wno-deprecated -Wno-unknown-pragmas)
    list(APPEND CPP_PLATFORM_LIBS util dl)
elseif(CMAKE_CXX_COMPILER MATCHES icpc)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -parallel -openmp -xSSSE3")
    add_definitions(-Wall -std=c99)
    add_definitions(-Wcheck)
    add_definitions(-wd1125) # virtual override intended warning
    add_definitions(-wd1572) # remove floating-pointe equality warning.
    add_definitions(-wd2259) # remove non-pointer conversion
    add_definitions(-wd1418) # External function with no prior declaration.
    add_definitions(-wd1419) # External declaration in primary source file.
    add_definitions(-wd383)  # Value copied to temporary, reference to temporary used.
    #add_definitions(-wd444)  # Destructor for base class not virtual.
    add_definitions(-wd981)  # Operands are evaluated in unspecified order.
    add_definitions(-wd177)  # Variable declared by never referenced.
    add_definitions(-ww111)  # Promote remark 111 to warning.
    add_definitions(-ww1572) # Promote remark 1572 to warning.
else(CMAKE_COMPILER_IS_GNUCXX)
    # use defaults (and pray it works...)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DNDEBUG")
endif(CMAKE_COMPILER_IS_GNUCXX)

if(APPLE)
    add_definitions(-DDARWIN)
endif(APPLE)




