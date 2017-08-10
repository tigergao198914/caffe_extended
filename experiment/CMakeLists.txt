# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set (CMAKE_CXX_STANDARD 11)

# target
add_executable(experiment ${srcs} ${Resources})
target_link_libraries(experiment ${Caffe_LINK} )

# set back RUNTIME_OUTPUT_DIRECTORY
caffe_set_runtime_directory(experiment "${PROJECT_BINARY_DIR}/experiment")
#caffe_set_solution_folder(visualization visualization)

# Install
install(TARGETS experiment DESTINATION ${CMAKE_INSTALL_BINDIR})
message("CMAKE_INSTALL_BINDIR:${CMAKE_INSTALL_BINDIR}")

