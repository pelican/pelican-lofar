set(lofarTest_src
    src/libTest.cpp
    src/LofarChunkerTest.cpp
)

set(lofarTestLib_src
)

add_library(lofarTestLib ${lofarTestLib_src})

add_executable(lofarTest ${lofarTest_src})

target_link_libraries(lofarTest
    lofarTestLib
    ${SUBPACKAGE_LIBRARIES}
    ${CPPUNIT_LIBRARIES}
)

add_test(libLofarTest libLofarTest)
