cmake_minimum_required(VERSION 2.8)

project( PedestrianDetection )

find_package( OpenCV REQUIRED )
# find_package( LibXml2 REQUIRED )
# include_directories(${LIBXML2_INCLUDE_DIRS})
# find_package( Armadillo REQUIRED )
# include_directories(${ARMADILLO_INCLUDE_DIRS})
#
# find_package( Boost REQUIRED )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package( MLPACK REQUIRED )
include_directories(${MLPACK_INCLUDE_DIRS})
set(LIBS ${LIBS} ${MLPACK_LIBRARIES})

find_package( Boost 1.54 COMPONENTS serialization system filesystem regex REQUIRED  )
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

find_package(Armadillo 2.4.2 REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lmlpack -lboost_system")

#add_executable( test testClient.cxx descriptorLBPH.cxx
#                        pedestrianRecognizer.cxx )
#target_link_libraries( test ${OpenCV_LIBS} )

add_executable( train trainingClient.cxx descriptorLBPH.cxx
                        pedestrianRecognizer.cxx )
target_link_libraries( train ${OpenCV_LIBS} ${LIBS} ${Boost_LIBRARIES})

add_executable( testWindow testWindow.cxx descriptorLBPH.cxx
                        pedestrianRecognizer.cxx )
target_link_libraries( testWindow ${OpenCV_LIBS}
                                    ${LIBS}
                                    ${Boost_LIBRARIES}
                                    ${ARMADILLO_LIBRARIES})

add_executable( testImage testImage.cxx pedestrianDetector.cxx
                    descriptorLBPH.cxx pedestrianRecognizer.cxx )
target_link_libraries( testImage ${OpenCV_LIBS}
                                    ${LIBS}
                                    ${Boost_LIBRARIES}
                                    ${ARMADILLO_LIBRARIES})


#add_executable( createNegative createNegativeCannonical.cxx )
#target_link_libraries( createNegative ${OpenCV_LIBS} )
