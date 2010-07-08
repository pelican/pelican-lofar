# CMAKE for Viewer module.
find_package(Qwt5 REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}
    ${QT_INCLUDE_DIR}
    ${QT_QTGUI_INCLUDE_DIR}
    ${Qwt5_INCLUDE_DIR}
)

set(widget_lib_UI
    forms/SubbandSpectraViewer.ui
)


# --- widget library
set(widget_lib_src
    src/LofarDataViewer.cpp
    src/DataViewer.cpp
    src/DataBlobWidget.cpp
    src/PlotWidget.cpp
    src/PlotPicker.cpp
    src/SubbandSpectrumWidget.cpp
)

set(widget_moc_headers
    LofarDataViewer.h
    DataViewer.h
    DataBlobWidget.h
    PlotWidget.h
    DataBlobWidget.h
    SubbandSpectrumWidget.h
)

QT4_WRAP_CPP(widget_moc_src ${widget_moc_headers})
qt4_wrap_ui(widget_lib_UI_H ${widget_lib_UI})

set(externals
    ${PELICAN_LIBRARY}
    pelican-lofar
    ${QT_QTGUI_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${QT_QTCORE_LIBRARY}
    ${Qwt5_Qt4_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

add_library(lofarWidgets
    ${widget_lib_src}
    ${widget_lib_UI_H}
    ${widget_moc_src}
)

target_link_libraries(lofarWidgets
    ${externals}
)

# --- main executable
add_executable(lofarDataViewer main.cpp)
target_link_libraries(lofarDataViewer lofarWidgets ${externals})

add_subdirectory(test)
