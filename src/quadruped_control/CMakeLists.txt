cmake_minimum_required(VERSION 3.16)
project(quadruped_control CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)

add_library(${PROJECT_NAME} STATIC
  src/kinematics.cpp
  src/estimator.cpp
  src/mlp.cpp
  src/policy.cpp
  src/config.cpp
  src/safety.cpp
  src/pose_generator.cpp
  src/controller.cpp
)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen ${YAML_CPP_LIBRARIES})
# The headers of the yaml-cpp that was found, not whichever /usr/include has (images with two of them).
# (Ubuntu 20.04's yaml-cpp 0.6 config points one level too high; skip a directory that is not there.)
if(YAML_CPP_INCLUDE_DIR)
  get_filename_component(YAML_CPP_INCLUDE_REAL ${YAML_CPP_INCLUDE_DIR} REALPATH)
  if(EXISTS ${YAML_CPP_INCLUDE_REAL}/yaml-cpp/yaml.h)
    target_include_directories(${PROJECT_NAME} PUBLIC ${YAML_CPP_INCLUDE_REAL})
  endif()
endif()

pybind11_add_module(_core python/bindings.cpp)
target_link_libraries(_core PRIVATE ${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR python/${PROJECT_NAME})
install(TARGETS _core DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME} ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(PROGRAMS test/estimator_parity.py test/policy_parity.py test/pipeline_parity.py scripts/export_policy.py DESTINATION lib/${PROJECT_NAME})
install(FILES test/reference.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY test/reference DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 yaml-cpp)
ament_package()
