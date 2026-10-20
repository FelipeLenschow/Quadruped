# FAST-LIO in C++17, with no change to the upstream submodule (third_party/FAST_LIO).
#
# FAST-LIO forces C++14 (add_compile_options(-std=c++14), CMAKE_CXX_STANDARD 14, -std=c++14 in CMAKE_CXX_FLAGS).
# ROS 2 Humble's rclcpp needs C++17 (std::variant). It builds only where CMake puts rclcpp's -std=c++17 last on
# the compile line (the PC); on the Jetson the C++14 wins and rclcpp's headers fail.
#
# colcon.meta (repo root) gives this file to fast_lio's configure as CMAKE_PROJECT_fast_lio_INCLUDE, so it runs
# at FAST-LIO's project(). It defers a call to the end of FAST-LIO's CMakeLists, which then removes every
# -std= flag and sets C++17 on its targets.

if(CMAKE_VERSION VERSION_LESS 3.19)
  message(FATAL_ERROR "fast_lio_cxx17.cmake needs CMake 3.19 or newer (cmake_language(DEFER)), not ${CMAKE_VERSION}")
endif()

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
  message(STATUS "fast_lio_cxx17.cmake: C++17 for ${changed}")
endfunction()

cmake_language(DEFER CALL _fast_lio_cxx17)
