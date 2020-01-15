cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolo-sort-app)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(Torch_DIR /home/hkuit155/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

aux_source_directory(. DIR_SRCS)

add_executable(yolo-sort-app ${DIR_SRCS})
target_link_libraries(yolo-sort-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET yolo-sort-app PROPERTY CXX_STANDARD 11)
