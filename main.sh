#!/bin/sh 


module purge
module load gcc cmake 
module load openmpi/4.0.1/gcc/8.2.0-2wc6vws


mkdir -p build
cd build 
cmake -Dtesting=ON .. 
cmake --build .
cd ..
