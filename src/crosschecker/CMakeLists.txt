include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set( crosschecker_src
     src/CorrelatingBuffer.cpp
     src/CorrelatedBufferManager.cpp
     src/CorrelationCheckModule.cpp
)
set( crosschecker_moc_headers
     CorrelatingBuffer.h
     CorrelatedBufferManager.h
     CorrelationCheckModule.h
)

# MOC Compile headers.
QT4_WRAP_CPP(crosschecker_src ${crosschecker_moc_headers})
add_library(crosschecker SHARED ${crosschecker_src} )
set_target_properties(pelican-lofar PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(pelican-lofar PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
install(TARGETS crosschecker DESTINATION ${LIBRARY_INSTALL_DIR})

set(crosschecker_externals
    ${PELICAN_LIBRARY}
    pelican-lofar
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${LAPACK_LIBRARIES}
)
target_link_libraries(crosschecker ${crosschecker-lofar_externals})

# --- main executable
add_executable(dmcrosscheck main.cpp)
target_link_libraries(dmcrosscheck crosschecker ${crosschecker_externals})

#add_subdirectory(test)
