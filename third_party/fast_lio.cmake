# FAST-LIO's build with two changes and no change to the upstream submodule (third_party/FAST_LIO).
#
# colcon.meta (repo root) gives this file to fast_lio's configure as CMAKE_PROJECT_fast_lio_INCLUDE, so it runs
# at FAST-LIO's project(). It defers a call to the end of FAST-LIO's CMakeLists, which makes both changes.
#
# 1. C++17. FAST-LIO forces C++14 (add_compile_options(-std=c++14), CMAKE_CXX_STANDARD 14, -std=c++14 in
#    CMAKE_CXX_FLAGS). ROS 2 Humble's rclcpp needs C++17 (std::variant). It builds only where CMake puts rclcpp's
#    -std=c++17 last on the compile line (the PC); on the Jetson the C++14 wins and rclcpp's headers fail. The
#    deferred call removes every -std= flag and sets C++17 on FAST-LIO's targets.
#
# 2. A level map. FAST-LIO takes the IMU's start attitude as its world frame, so a robot that starts pitched or
#    rolled leans the whole map and path by that much. fast_lio_level.patch (next to this file) starts the world
#    frame level instead. Here src/ is copied to the build folder and patched, and the deferred call points
#    fastlio_mapping's sources at the copy (whose quoted includes then find the patched IMU_Processing.hpp).

if(CMAKE_VERSION VERSION_LESS 3.19)
  message(FATAL_ERROR "fast_lio.cmake needs CMake 3.19 or newer (cmake_language(DEFER)), not ${CMAKE_VERSION}")
endif()

set(_fast_lio_patched "${CMAKE_CURRENT_BINARY_DIR}/patched")
file(REMOVE_RECURSE "${_fast_lio_patched}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src" DESTINATION "${_fast_lio_patched}")
execute_process(COMMAND patch -p1 --forward -d "${_fast_lio_patched}" -i "${CMAKE_CURRENT_LIST_DIR}/fast_lio_level.patch"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "fast_lio.cmake: fast_lio_level.patch does not apply to FAST-LIO's src/:\n${output}")
endif()
file(GLOB _fast_lio_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_LIST_DIR}/fast_lio_level.patch" ${_fast_lio_sources})

function(_fast_lio_cxx17)
  string(REGEX REPLACE "-std=[a-z+0-9]+" "" flags "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${flags}" PARENT_SCOPE)

  get_property(options DIRECTORY PROPERTY COMPILE_OPTIONS)
  list(FILTER options EXCLUDE REGEX "^-std=")
  set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "${options}")

  get_property(targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  set(changed "")
  foreach(target IN LISTS targets)
    get_target_property(type ${target} TYPE)
    if(NOT type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|STATIC_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY)$")
      continue()
    endif()
    get_target_property(options ${target} COMPILE_OPTIONS)
    if(options)
      list(FILTER options EXCLUDE REGEX "^-std=")
      set_target_properties(${target} PROPERTIES COMPILE_OPTIONS "${options}")
    endif()
    set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    list(APPEND changed ${target})
  endforeach()
  list(JOIN changed ", " changed)
  message(STATUS "fast_lio.cmake: C++17 for ${changed}")
endfunction()

function(_fast_lio_level)
  get_target_property(sources fastlio_mapping SOURCES)
  list(TRANSFORM sources REPLACE "^src/" "${_fast_lio_patched}/src/")
  set_target_properties(fastlio_mapping PROPERTIES SOURCES "${sources}")
  message(STATUS "fast_lio.cmake: level map, fastlio_mapping from ${_fast_lio_patched}/src")
endfunction()

cmake_language(DEFER CALL _fast_lio_cxx17)
cmake_language(DEFER CALL _fast_lio_level)
