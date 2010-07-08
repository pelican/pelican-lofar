include_directories(${Qwt5_INCLUDE_DIR})

# Command Line options testing.
qt4_wrap_cpp(plotWidgetTest_MOC PlotWidgetTest.h)
add_executable(plotWidgetTest ${plotWidgetTest_MOC})
target_link_libraries(plotWidgetTest
    lofarWidgets
    ${Qwt5_Qt4_LIBRARY}
    ${QT_QTTEST_LIBRARY}
    ${QT_QTGUI_LIBRARY}
)

