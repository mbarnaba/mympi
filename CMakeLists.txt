cmake_minimum_required( VERSION 3.0 )

project( mympi LANGUAGES C CXX )

foreach( language IN ITEMS C CXX ) 
    set( CMAKE_${language}_STANDARD 11 )
    set( CMAKE_${language}_EXTENSIONS OFF )
endforeach()

add_compile_options(
    -Wall -Wextra -pedantic -Werror -fmax-errors=1
)


option( verbose_make "make make verbose" ON )
option( testing "enable_testing" ON )

if( verbose_make ) 
    message( STATUS "making make verbose" )
    set( CMAKE_VERBOSE_MAKEFILE true )
endif()



set( target mympic )

find_package( MPI REQUIRED )

add_library( 
    ${target} SHARED 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mympi.c 
)
target_include_directories( 
    ${target} PUBLIC 
    ${MPI_C_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
target_link_libraries( 
    ${target} PRIVATE
    ${MPI_LIBRARIES}
)


set( target mympicpp )

add_library(
    ${target} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mympi.cpp
)
target_include_directories( 
    ${target} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)
target_link_libraries( 
    ${target} PUBLIC 
    mympic
)


if (testing) 
    message( STATUS "enabling testing" )
    enable_testing()

    set( target ctest )
    set( ltarget ${CMAKE_CURRENT_BINARY_DIR}/${target} )
    add_executable( ${target} ${CMAKE_CURRENT_SOURCE_DIR}/tests/main.c )
    target_link_libraries( ${target} PRIVATE mympic )

    set( tname ${target} )
    add_test( 
        ${tname} 
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/ctest.sh ${ltarget} 
    )
endif()
