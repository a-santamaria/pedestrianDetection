cmake_minimum_required(VERSION 2.8)

project( PedestrianDetection )

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#add_executable( test testClient.cxx descriptorLBPH.cxx
#                        pedestrianRecognizer.cxx )
#target_link_libraries( test ${OpenCV_LIBS} )

add_executable( train trainingClient.cxx descriptorLBPH.cxx
                        pedestrianRecognizer.cxx )
target_link_libraries( train ${OpenCV_LIBS} )

add_executable( testWindow testWindow.cxx descriptorLBPH.cxx
                        pedestrianRecognizer.cxx )
target_link_libraries( testWindow ${OpenCV_LIBS} )


#add_executable( createNegative createNegativeCannonical.cxx )
#target_link_libraries( createNegative ${OpenCV_LIBS} )
