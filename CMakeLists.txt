
cmake_minimum_required(VERSION 3.4.1)
set (CMAKE_C_STANDARD 99)
project(pro_vad)
SET(FVAD /home/wanghaibo/docker_images/code/wenet-build/runtime/server/x86/vad/lib/libfvad.so)
SET(SNDFILE /home/wanghaibo/docker_images/code/wenet-build/runtime/server/x86/vad/lib/libsndfile.so)
add_executable(test_vad ${PROJECT_SOURCE_DIR}/test/fvadwav.c)
link_directories(${PROJECT_SOURCE_DIR}/lib)
include_directories(${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_vad ${FVAD} ${SNDFILE}) # 链接库到执行文件上







