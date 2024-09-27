cmake_minimum_required(VERSION 3.8)
project(yurs_arm_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  controller_interface
  generate_parameter_library
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  std_msgs
  moveit_msgs
  sensor_msgs
  trajectory_msgs
)

find_package(ament_cmake REQUIRED)
find_package(backward_ros REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

#adding executable
add_executable(doublePID src/doublePID.cpp)
ament_target_dependencies(doublePID rclcpp std_msgs trajectory_msgs sensor_msgs moveit_msgs) 

#generate parameters
generate_parameter_library(
  yurs_arm_controller_parameters
  src/yurs_arm_controller_parameters.yaml
)

add_library(yurs_arm_controller SHARED
  src/yurs_arm_controllers_base.cpp
  src/yurs_arm_controller.cpp
)

target_compile_features(yurs_arm_controller PUBLIC cxx_std_17)
target_include_directories(yurs_arm_controller PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/yurs_arm_controller>
)

target_link_libraries(yurs_arm_controller PUBLIC
  yurs_arm_controller_parameters
)

ament_target_dependencies(yurs_arm_controller PUBLIC ${THIS_PACKAGE_INCLUDE_DEPENDS})
# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(yurs_arm_controller PRIVATE "YURS_ARM_CONTROLLER_BUILDING_DLL")
pluginlib_export_plugin_description_file(controller_interface yurs_arm_plugin.xml)

if(BUILD_TESTING)
  find_package(ament_cmake_gmock REQUIRED)
  find_package(controller_manager REQUIRED)
  find_package(ros2_control_test_assets REQUIRED)

  ament_add_gmock(test_load_yurs_arm_controller
    test/test_load_yurs_arm_controller.cpp
  )
  target_link_libraries(test_load_yurs_arm_controller
    yurs_arm_controller
  )
  ament_target_dependencies(test_load_yurs_arm_controller
    controller_manager
    hardware_interface
    ros2_control_test_assets
  )
  ament_add_gmock(test_yurs_arm_controller
    test/test_yurs_arm_controller.cpp
  )
  target_link_libraries(test_yurs_arm_controller
    yurs_arm_controller
  )
endif()

install(
  DIRECTORY include/
  DESTINATION include/yurs_arm_controller
)
install(
  TARGETS
    yurs_arm_controller
    yurs_arm_controller_parameters
  EXPORT export_yurs_arm_controller
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

ament_export_targets(export_yurs_arm_controller HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})
ament_package()