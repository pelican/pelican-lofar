set(lofarTest_src
    src/libTest.cpp
    src/LofarChunkerTest.cpp
)

add_executable(lofarTest ${lofarTest_src})

target_link_libraries(lofarTest
    ${SUBPACKAGE_LIBRARIES}
    ${CPPUNIT_LIBRARIES}
)

#add_test(libLofarTest libLofarTest)
