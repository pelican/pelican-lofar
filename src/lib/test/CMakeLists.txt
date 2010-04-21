include_directories(${CMAKE_CURRENT_SOURCE_DIR} ..)
set(lofarTest_src
    src/libTest.cpp
    src/LofarChunkerTest.cpp
    src/DataStreamingTest.cpp
)

set(lofarTestLib_src
    src/LofarDataGenerator.cpp
)

add_library(lofarTestLib ${lofarTestLib_src})

add_executable(lofarTest ${lofarTest_src})

target_link_libraries(lofarTest
    lofarTestLib
    ${SUBPACKAGE_LIBRARIES}
    ${CPPUNIT_LIBRARIES}
)

add_test(libLofarTest libLofarTest)
