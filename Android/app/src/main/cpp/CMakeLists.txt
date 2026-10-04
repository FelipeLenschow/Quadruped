cmake_minimum_required(VERSION 3.22)
project(roslink CXX)

set(CMAKE_CXX_STANDARD 17)

# Built by ../../../../build_fastdds.sh
set(FASTDDS_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../../../../native/fastdds/${ANDROID_ABI})
set(fastcdr_DIR ${FASTDDS_PREFIX}/lib/cmake/fastcdr)
set(foonathan_memory_DIR ${FASTDDS_PREFIX}/lib/foonathan_memory/cmake)
set(fastrtps_DIR ${FASTDDS_PREFIX}/share/fastrtps/cmake)
find_package(fastcdr REQUIRED)
find_package(foonathan_memory REQUIRED)
find_package(fastrtps REQUIRED)

add_library(roslink SHARED jni.cpp ros_link.cpp)
target_link_libraries(roslink fastrtps fastcdr log)
