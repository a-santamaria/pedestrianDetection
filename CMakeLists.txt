cmake_minimum_required(VERSION 2.8)

project( PedestrianDetection )

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable( test testCleint.cxx )
target_link_libraries( test ${OpenCV_LIBS} )
