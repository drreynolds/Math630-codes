###################################################################
# CMakeLists file for matrix-vector product example.  To use, type
# 'cmake .' at the command line, and it will replace the 'Makefile'
# in this directory.
#
# Daniel R. Reynolds
# Math 630 @ UMBC
# Spring 2026
###################################################################

# project and version information
cmake_minimum_required (VERSION 3.10)
project (matvec)

# require at least C++ 11
set (CMAKE_CXX_STANDARD 11)

# Add executable called "driver.exe" that is built from the source file
# "driver.cpp".
add_executable (driver.exe driver.cpp)
