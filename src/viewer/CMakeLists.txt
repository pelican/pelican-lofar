include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${QT_INCLUDE_DIR})

# --- widget library
set( widget_lib_src 
     src/LofarDataViewer.cpp
     src/DataViewer.cpp
     src/DataBlobWidget.cpp
   )

set( widget_moc_headers
    LofarDataViewer.h
    DataViewer.h
    DataBlobWidget.h
    )

QT4_WRAP_CPP(widget_moc ${widget_moc_headers})

set(externals
    ${PELICAN_LIBRARY}
    ${QT_QTGUI_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    )

add_library( lofarWidgets ${widget_lib_src} ${widget_moc} )

# --- main executable
add_executable(lofarDataViewer main.cpp)
target_link_libraries(lofarDataViewer lofarWidgets pelican-lofar ${externals} )
