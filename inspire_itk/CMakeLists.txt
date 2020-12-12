cmake_minimum_required(VERSION 3.5)
project(INSPIRE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(InspireRegister source/programs/registerMain.cpp ${HEADERS})

add_executable(InspireTransform source/programs/transformMain.cpp)

add_executable(InspireTransformLandmarks source/programs/transformLandmarksMain.cpp)

add_executable(InspireEvaluateRegistration source/programs/evaluateRegistrationMain.cpp)

add_executable(InspireTestSuite source/tests/testSuite.cpp)

add_executable(InspireRandomBSpline source/programs/synthDeformImageMain.cpp)

target_link_libraries(InspireRegister ${ITK_LIBRARIES})

target_link_libraries(InspireTransform ${ITK_LIBRARIES})

target_link_libraries(InspireTransformLandmarks ${ITK_LIBRARIES})

target_link_libraries(InspireEvaluateRegistration ${ITK_LIBRARIES})

target_link_libraries(InspireTestSuite ${ITK_LIBRARIES})

target_link_libraries(InspireRandomBSpline ${ITK_LIBRARIES})
