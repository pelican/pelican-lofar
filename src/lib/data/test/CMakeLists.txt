# Create test binary and add it to the cmake test framework.
# ------------------------------------------------------------------------------
set(lofarDataTest_src
    src/lofarDataTest.cpp
    src/MatrixTest.cpp
    src/LofarDataCubeTest.cpp
    src/CubeTest.cpp
)

add_executable(lofarDataTest ${lofarDataTest_src})

target_link_libraries(lofarDataTest
    ${CPPUNIT_LIBRARIES}
)

#add_test(lofarTest lofarDataTest)

