cmake_minimum_required(VERSION 3.18)
project(LocalEdgeAssistant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find or specify path to ONNX Runtime C++ SDK
find_package(ONNXRuntime REQUIRED)

include_directories(src/include ${ONNXRUNTIME_INCLUDE_DIRS})

add_executable(edge_assistant
    src/main.cpp
    src/engine.cpp
)

target_link_libraries(edge_assistant PRIVATE ${ONNXRUNTIME_LIBRARIES})
