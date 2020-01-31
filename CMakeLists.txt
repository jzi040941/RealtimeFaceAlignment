cmake_minimum_required(VERSION 3.12)
project(ch3_face)

set(CMAKE_CXX_COMPILER /usr/bin/g++-6)
set(CMAKE_CXX_STANDARD 11)

#set(OpenCV_DIR "" CACHE PATH "OpenCV 4 build Directory")
#find_package(OpenCV 4 REQUIRED core face highgui calib3d)

#find_package(Boost REQUIRED system)

set(INCL)
set(LINKLIBS)

list(APPEND INCL
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv4
)

list(APPEND LINKLIBS 
  ${PROJECT_SOURCE_DIR}/lib/opencv4/libopencv_core.so.4.2.0
  ${PROJECT_SOURCE_DIR}/lib/opencv4/libopencv_highgui.so.4.2.0
  ${PROJECT_SOURCE_DIR}/lib/opencv4/libopencv_imgproc.so.4.2.0
  ${PROJECT_SOURCE_DIR}/lib/opencv4/libopencv_objdetect.so.4.2.0
  ${PROJECT_SOURCE_DIR}/lib/opencv4/libopencv_videoio.so.4.2.0
  ${PROJECT_SOURCE_DIR}/lib/opencv4/libopencv_dnn.so.4.2.0
  ${PROJECT_SOURCE_DIR}/lib/opencv4/libopencv_face.so.4.2.0
  ${PROJECT_SOURCE_DIR}/lib/opencv4/libopencv_imgcodecs.so.4.2.0
  ${PROJECT_SOURCE_DIR}/lib/opencv4/libopencv_calib3d.so.4.2.0
  )

include_directories(${INCL})

add_executable(ch3_face main.cpp)

target_link_libraries(ch3_face ${LINKLIBS})
