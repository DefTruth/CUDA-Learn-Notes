PROJECT(staticlib_demo C CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

include_directories(${PROJECT_SOURCE_DIR})

set(LIB_ADD_SRCS
        ${PROJECT_SOURCE_DIR}/liba/func.cc
        ${PROJECT_SOURCE_DIR}/liba/funca.cc
        ${PROJECT_SOURCE_DIR}/liba/global.cc
        ${PROJECT_SOURCE_DIR}/libb/func.cc
        ${PROJECT_SOURCE_DIR}/libb/funcb.cc)

set(LIB_OTHER_ADD_SRCS
        ${PROJECT_SOURCE_DIR}/libc/funcc.cc)

add_library(addfunc_static STATIC ${LIB_ADD_SRCS}) # 编译静态库 
add_library(addfunc_shared SHARED ${LIB_ADD_SRCS}) # 编译动态库
add_library(other_addfunc_static_link_static STATIC ${LIB_OTHER_ADD_SRCS})
add_library(other_addfunc_static_link_shared STATIC ${LIB_OTHER_ADD_SRCS})
add_library(other_addfunc_shared_link_static SHARED ${LIB_OTHER_ADD_SRCS})
add_library(other_addfunc_shared_link_shared SHARED ${LIB_OTHER_ADD_SRCS})

target_link_libraries(other_addfunc_static_link_static addfunc_static) # 编译静态库链接静态库
target_link_libraries(other_addfunc_static_link_shared addfunc_shared) # 编译静态库链接动态库
target_link_libraries(other_addfunc_shared_link_static addfunc_static) # 编译动态库链接静态库
target_link_libraries(other_addfunc_shared_link_shared addfunc_shared) # 编译动态库链接动态库

add_executable(test_static ${PROJECT_SOURCE_DIR}/test.cc)
add_executable(test_shared ${PROJECT_SOURCE_DIR}/test.cc)
target_link_libraries(test_static addfunc_static)
set_target_properties(test_static PROPERTIES LINK_FLAGS 
                      "-Wl,--whole-archive libaddfunc_static.a -Wl,-no-whole-archive") 

target_link_libraries(test_shared addfunc_shared)

add_executable(test_a_inst_static ${PROJECT_SOURCE_DIR}/test_a_inst.cc)
add_executable(test_a_inst_shared ${PROJECT_SOURCE_DIR}/test_a_inst.cc)
target_link_libraries(test_a_inst_static addfunc_static)
target_link_libraries(test_a_inst_shared addfunc_shared)

