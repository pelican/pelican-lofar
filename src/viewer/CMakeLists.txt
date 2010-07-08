# CMAKE for Viewer module.


find_package(Qwt5 REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${QT_INCLUDE_DIR}
    ${QT_QTGUI_INCLUDE_DIR}
    ${Qwt5_INCLUDE_DIR}
)

# --- widget library
set(widget_lib_src
     src/LofarDataViewer.cpp
     src/DataViewer.cpp
     src/DataBlobWidget.cpp
     src/PlotWidget.cpp
     src/PlotPicker.cpp
     src/ThreadedBlobClient.cpp
)

set(widget_moc_headers
    LofarDataViewer.h
    DataViewer.h
    DataBlobWidget.h
    PlotWidget.h
    ThreadedBlobClient.h
)

QT4_WRAP_CPP(widget_moc ${widget_moc_headers})

set(externals
    ${PELICAN_LIBRARY}
    ${QT_QTGUI_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${Qwt5_Qt4_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

add_library(lofarWidgets ${widget_lib_src} ${widget_moc})

target_link_libraries(lofarWidgets
    ${externals}
)

# --- main executable
add_executable(lofarDataViewer main.cpp)
target_link_libraries(lofarDataViewer lofarWidgets pelican-lofar ${externals})

add_subdirectory(test)
