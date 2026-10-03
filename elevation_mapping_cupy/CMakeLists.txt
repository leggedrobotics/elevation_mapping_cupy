cmake_minimum_required(VERSION 3.8)
project(elevation_mapping_cupy)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(elevation_map_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

set(dependencies
  cv_bridge
  elevation_map_msgs
  message_filters
  rclpy
  std_msgs
  sensor_msgs
  geometry_msgs
  grid_map_msgs
  tf2_ros
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/elevation_mapping_node.py
  scripts/synthetic_pointcloud_tf_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

# Install test files for integration tests
install(
  DIRECTORY test
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_ros REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)

  # Unit tests (run via pytest, no ROS dependencies)
  # These are the PRIMARY regression tests for the axis-swap bug
  # Note: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 disables the launch_testing pytest plugin
  # which would otherwise try to import the package (triggering cupy import errors)
  ament_add_pytest_test(test_map_shifting
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_map_shifting.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_map_services
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_map_services.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_gridmap_layout
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_gridmap_layout.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_parameter
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_parameter.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_repo_config_sanity
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_repo_config_sanity.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_inpainting_plugin
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_inpainting_plugin.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_kernel_compile_smoke
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_kernel_compile_smoke.py
    TIMEOUT 180
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_dilation_filter
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_dilation_filter.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_fusion_finalization
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_fusion_finalization.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_observation_age
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_observation_age.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_indexing_center
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_indexing_center.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_height_gate_translation_invariance
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_height_gate_translation_invariance.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_plugin_manager_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_plugin_manager_cache.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_pointcloud_parser
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/tests/test_pointcloud_parser.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )
  ament_add_pytest_test(test_semantic_launch_descriptions
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_semantic_launch_descriptions.py
    TIMEOUT 120
    ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  )

  # Define isolated launch test function for unique ROS_DOMAIN_ID per test
  # See: https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Testing/Integration.html
  function(add_ros_isolated_launch_test path)
    set(RUNNER "${ament_cmake_ros_DIR}/run_test_isolated.py")
    add_launch_test("${path}" RUNNER "${RUNNER}" ${ARGN})
  endfunction()

  # Integration test (launches ROS nodes)
  # DDS fixes:
  #   FASTDDS_BUILTIN_TRANSPORTS=UDPv4: Avoid FastDDS shared memory discovery issues
  #   See: https://answers.ros.org/question/407025/
  # SKIP_DDS_FLAKES=1: In CI, skip (not fail) on known DDS discovery issues.
  # Locally, tests fail loudly to catch real regressions.
  add_ros_isolated_launch_test(test/test_tf_gridmap_integration.py
    TIMEOUT 180
    ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
  )

  add_ros_isolated_launch_test(test/test_map_save_load_services.py
    TIMEOUT 240
    ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
  )

  # Smoke-test the actual golden-path launch file (without RViz).
  add_ros_isolated_launch_test(test/test_synthetic_demo_launch.py
    TIMEOUT 180
    ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
  )
endif()

ament_export_dependencies(${dependencies})
ament_package()
