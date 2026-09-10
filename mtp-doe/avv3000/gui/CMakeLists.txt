# AVV-3000 QC panel (rung r_gui) — Qt6 Widgets + QtTest, offscreen.
# Configure with -DCMAKE_PREFIX_PATH=$(brew --prefix)/opt/qt
cmake_minimum_required(VERSION 3.16)
project(avv3000_gui C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Test)

add_executable(t_gui_unit
    t_gui_unit.cpp
    qcpanel.cpp
    qc_rate.c)
target_link_libraries(t_gui_unit PRIVATE Qt6::Widgets Qt6::Test)
target_compile_options(t_gui_unit PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

enable_testing()
add_test(NAME t_gui_unit COMMAND t_gui_unit)
set_tests_properties(t_gui_unit PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
