include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set( crosschecker_src
     src/CorrelatingBuffer.cpp
)
set( crosschecker_moc_headers
     CorrelatingBuffer.h
)

# MOC Compile headers.
QT4_WRAP_CPP(crosschecker ${crosschecker_moc_headers})
add_library(crosschecker SHARED ${crosschecker_src} )
set_target_properties(pelican-lofar PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(pelican-lofar PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
install(TARGETS crosschecker DESTINATION ${LIBRARY_INSTALL_DIR})

set(crosschecker_externals
    ${FFTW3_FFTW_LIBRARY}
    ${FFTW3_FFTWF_LIBRARY}
    ${LAPACK_LIBRARIES}
)
target_link_libraries(crosschecker ${crosschecker-lofar_externals})

#add_subdirectory(test)
