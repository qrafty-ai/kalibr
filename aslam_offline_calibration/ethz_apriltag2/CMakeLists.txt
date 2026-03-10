cmake_minimum_required(VERSION 3.28.0)
project(ethz_apriltag2
  VERSION 0.0.1
  DESCRIPTION "ETHZ AprilTag2 library"
  LANGUAGES CXX
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_path(V4L2_INCLUDE_DIR
  NAMES libv4l2.h
  PATHS /usr/include /usr/local/include
)
find_library(V4L2_LIBRARY
  NAMES v4l2
  PATHS /usr/lib64 /usr/lib /lib64 /lib
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


add_library(
  ${PROJECT_NAME}
  src/Edge.cc
  src/FloatImage.cc
  src/GLine2D.cc
  src/GLineSegment2D.cc
  src/Gaussian.cc
  src/GrayModel.cc
  src/Homography33.cc
  src/MathUtil.cc
  src/Quad.cc
  src/Segment.cc
  src/TagDetection.cc
  src/TagDetector.cc
  src/TagFamily.cc
  src/UnionFindSimple.cc
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${OpenCV_INCLUDE_DIRS}
)

#library
target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

#demo
if(NOT APPLE)
  if(NOT V4L2_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find libv4l2.h. Set V4L2_INCLUDE_DIR or install the v4l2 development headers.")
  endif()
  if(NOT V4L2_LIBRARY)
    message(FATAL_ERROR "Could not find libv4l2. Set V4L2_LIBRARY or install the v4l2 runtime library.")
  endif()

  add_executable(apriltags_demo src/example/apriltags_demo.cpp src/example/Serial.cpp)
  target_include_directories(apriltags_demo PRIVATE ${V4L2_INCLUDE_DIR})
  target_compile_options(apriltags_demo PRIVATE "SHELL:-isystem ${V4L2_INCLUDE_DIR}")
  target_link_libraries(apriltags_demo ${PROJECT_NAME} ${V4L2_LIBRARY})
endif()

target_compile_options(
  ${PROJECT_NAME}
  PUBLIC
  "-Wall"
  "-Wextra"
)

target_compile_features(
  ${PROJECT_NAME}
  PUBLIC
  cxx_std_14
)

set_target_properties(
  ${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON
)

set(INSTALL_CMAKEDIR ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/cmake)

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${INSTALL_CMAKEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${INSTALL_CMAKEDIR})
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${INSTALL_CMAKEDIR})
