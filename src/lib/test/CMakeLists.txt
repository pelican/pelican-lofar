include_directories(${CMAKE_CURRENT_SOURCE_DIR} ..)

set(lofarTest_src
    src/libTest.cpp
    src/LofarChunkerTest.cpp
    src/DataStreamingTest.cpp
    src/TimeStreamDataTest.cpp
    src/ChannelisedStreamDataTest.cpp
    src/AdapterTimeStreamTest.cpp
    src/PolyphaseCoefficientsTest.cpp
    src/ChanneliserPolyphaseTest.cpp
)

set(lofarTest_moc_headers )

set(lofarTestLib_src src/LofarUdpEmulator.cpp)

QT4_WRAP_CPP(lofarTest_moc_sources ${lofarTest_moc_headers})

add_library(lofarTestLib ${lofarTestLib_src} ${lofarTest_moc_sources})

add_executable(lofarTest ${lofarTest_src})

target_link_libraries(lofarTest
    lofarTestLib
    ${SUBPACKAGE_LIBRARIES}
    ${CPPUNIT_LIBRARIES}
)

add_test(lofarTest lofarTest)


