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

add_executable(real_driver
  src/main.cpp
  src/realtime.cpp
  src/robot_io.cpp
  src/control_loop.cpp
  src/driver_node.cpp
  src/json.cpp
)
target_include_directories(real_driver PRIVATE src)
target_link_libraries(real_driver quadruped_control::quadruped_control unitree_sdk2)
ament_target_dependencies(real_driver rclcpp std_msgs sensor_msgs nav_msgs geometry_msgs tf2_ros ament_index_cpp)
set_target_properties(real_driver PROPERTIES BUILD_RPATH ${UNITREE_DDS_LIB} INSTALL_RPATH ${UNITREE_DDS_LIB})
# RPATH, not RUNPATH: LD_LIBRARY_PATH must not swap in another libddsc (the robot image builds one into /usr/local/lib).
target_link_options(real_driver PRIVATE -Wl,--disable-new-dtags)

install(TARGETS real_driver DESTINATION lib/${PROJECT_NAME})
ament_package()
