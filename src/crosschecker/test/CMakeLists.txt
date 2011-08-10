SUBPACKAGE(crosscheckerTest crosschecker)
set(
   crosscheckerTest_src
src/crosscheckerTest.cpp
)
add_executable(crosscheckerTest ${crosscheckerTest_src} )
target_link_libraries(crosscheckerTest ${SUBPACKAGE_LIBRARIES} ${CPPUNIT_LIBRARIES})
add_test(crosscheckerTest crosscheckerTest )
