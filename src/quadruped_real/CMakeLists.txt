cmake_minimum_required(VERSION 3.16)
project(quadruped_real CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(quadruped_control REQUIRED)
find_package(Threads REQUIRED)

# Unitree SDK2 (third_party submodule): a static library and its own CycloneDDS. The DDS libraries are
# found through the RPATH, not installed: on LD_LIBRARY_PATH they would shadow ROS's own libddsc.
set(UNITREE_SDK ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/unitree_sdk2)
get_filename_component(UNITREE_SDK ${UNITREE_SDK} REALPATH)
set(UNITREE_DDS_LIB ${UNITREE_SDK}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR})
if(NOT EXISTS ${UNITREE_SDK}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunitree_sdk2.a)
  message(FATAL_ERROR "Unitree SDK2 not found at ${UNITREE_SDK}: git submodule update --init third_party/unitree_sdk2")
endif()
add_library(unitree_sdk2 STATIC IMPORTED)
set_target_properties(unitree_sdk2 PROPERTIES IMPORTED_LOCATION ${UNITREE_SDK}/lib/${CMAKE_SYSTEM_PROCESSOR}/libunitree_sdk2.a)
target_include_directories(unitree_sdk2 INTERFACE ${UNITREE_SDK}/include ${UNITREE_SDK}/thirdparty/include
                           ${UNITREE_SDK}/thirdparty/include/ddscxx)
target_link_libraries(unitree_sdk2 INTERFACE ${UNITREE_DDS_LIB}/libddscxx.so ${UNITREE_DDS_LIB}/libddsc.so Threads::Threads)

# Each executable gets the SDK's DDS libraries by RPATH, not RUNPATH: LD_LIBRARY_PATH must not swap in another
# libddsc. yaml-cpp and Eigen come through quadruped_control, so every one uses the yaml-cpp headers it was
# built with (an image with two yaml-cpp versions crashed on the mismatch).
function(sdk_executable name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE src)
  target_link_libraries(${name} unitree_sdk2 quadruped_control::quadruped_control)
  ament_target_dependencies(${name} rclcpp ament_index_cpp)
  set_target_properties(${name} PROPERTIES BUILD_RPATH "${UNITREE_DDS_LIB}" INSTALL_RPATH "${UNITREE_DDS_LIB}")
  target_link_options(${name} PRIVATE -Wl,--disable-new-dtags)
  install(TARGETS ${name} DESTINATION lib/${PROJECT_NAME})
endfunction()

sdk_executable(real_driver src/main.cpp src/paths.cpp src/realtime.cpp src/robot_io.cpp src/control_loop.cpp
               src/driver_node.cpp src/json.cpp)
ament_target_dependencies(real_driver std_msgs sensor_msgs nav_msgs geometry_msgs tf2_ros)

sdk_executable(real_sensors src/real_sensors.cpp src/paths.cpp)
ament_target_dependencies(real_sensors sensor_msgs geometry_msgs tf2_ros)

sdk_executable(go2_camera src/go2_camera.cpp src/paths.cpp)
ament_target_dependencies(go2_camera sensor_msgs)

sdk_executable(test_joints src/test_joints.cpp src/paths.cpp src/robot_io.cpp)

# mock_go2 (a MuJoCo Go2 on the SDK's topics, for running real_driver on a PC) needs MuJoCo's C library, which
# the mujoco Python wheel carries. Skipped where it is not installed.
find_package(Python3 COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
                OUTPUT_VARIABLE MUJOCO_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
file(GLOB MUJOCO_LIB "${MUJOCO_DIR}/libmujoco.so*")
if(MUJOCO_DIR AND MUJOCO_LIB AND EXISTS ${MUJOCO_DIR}/include/mujoco/mujoco.h)
  list(GET MUJOCO_LIB 0 MUJOCO_LIB)
  sdk_executable(mock_go2 test/mock_go2.cpp src/paths.cpp src/robot_io.cpp)
  target_include_directories(mock_go2 PRIVATE ${MUJOCO_DIR}/include)
  target_compile_definitions(mock_go2 PRIVATE MUJOCO_PLUGIN_DIR="${MUJOCO_DIR}/plugin")
  target_link_libraries(mock_go2 ${MUJOCO_LIB})
  set_target_properties(mock_go2 PROPERTIES BUILD_RPATH "${UNITREE_DDS_LIB};${MUJOCO_DIR}"
                                            INSTALL_RPATH "${UNITREE_DDS_LIB};${MUJOCO_DIR}")
  # Its window needs GLFW; without it, mock_go2 is headless.
  find_path(GLFW_INCLUDE GLFW/glfw3.h)
  find_library(GLFW_LIB glfw)
  if(GLFW_INCLUDE AND GLFW_LIB)
    target_include_directories(mock_go2 PRIVATE ${GLFW_INCLUDE})
    target_compile_definitions(mock_go2 PRIVATE MOCK_GO2_VIEWER)
    target_link_libraries(mock_go2 ${GLFW_LIB})
  else()
    message(STATUS "GLFW not found: mock_go2 has no window")
  endif()
else()
  message(STATUS "mujoco (Python wheel) not found: mock_go2 is not built")
endif()

ament_package()
