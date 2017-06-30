# Collect source files
file(GLOB_RECURSE srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_PREFIX_PATH
  /home/hosery/Qt5.6.1/5.6/gcc_64/lib/cmake/Qt5Core;
  /home/hosery/Qt5.6.1/5.6/gcc_64/lib/cmake/Qt5Gui;
  /home/hosery/Qt5.6.1/5.6/gcc_64/lib/cmake/Qt5Widgets;
  /home/hosery/Qt5.6.1/5.6/gcc_64/lib/cmake/Qt5Quick;
  /home/hosery/Qt5.6.1/5.6/gcc_64/lib/cmake/Qt5Qml;
  /usr/lib/ffmpeg;
  )

#add resource
file (GLOB Resources *.qrc )

message("${Resources}")

find_package(Qt5Widgets REQUIRED)
find_package(Qt5Quick REQUIRED)
find_package(Qt5Qml REQUIRED)

# target
add_executable(train_visualization ${srcs} ${Resources})
target_link_libraries(train_visualization ${Caffe_LINK} Qt5::Widgets Qt5::Quick Qt5::Qml)

# set back RUNTIME_OUTPUT_DIRECTORY
caffe_set_runtime_directory(train_visualization "${PROJECT_BINARY_DIR}/train_visualization")
#caffe_set_solution_folder(visualization visualization)

# Install
install(TARGETS train_visualization DESTINATION ${CMAKE_INSTALL_BINDIR})
message("CMAKE_INSTALL_BINDIR:${CMAKE_INSTALL_BINDIR}")

