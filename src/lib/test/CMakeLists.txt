set(
    libLofarTest_src
    src/libTest.cpp
    src/LofarChunkerTest.cpp
)
add_executable(lofarTest ${libLofarTest_src} )
message(STATUS "${SUBPACKAGE_LIBRARIES} ${CPPUNIT_LIBRARIES}")
target_link_libraries(lofarTest ${SUBPACKAGE_LIBRARIES} ${CPPUNIT_LIBRARIES} )
add_test(libLofarTest libLofarTest )
