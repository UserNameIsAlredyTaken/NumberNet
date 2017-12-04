cmake_minimum_required(VERSION 3.8)
project(NetJet C)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES
        src/main.c
        src/get_train_data.c
        src/net.c

        head/get_data.h
        head/net.h)
add_executable(NetJet ${SOURCE_FILES})