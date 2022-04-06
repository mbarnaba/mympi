#! /bin/bash 


otest() {
    local pattern="$1"
    
    [[ -z "$pattern" ]] && return 0
    grep -w "$pattern" "$ofile" && return 0
    
    echo "PATTERN NOT FOUND (should be) IN $ofile"
    echo "$pattern"
    exit 1
}

repeat() {
    local what="$1"
    local times="$2"
    
    local out=''
    for counter in $( seq $times ); do
        out="${out}${what}"
    done 
    echo "$out"
}


exe="$1"

module purge
module load gcc cmake 
module load openmpi/4.0.1/gcc/8.2.0-2wc6vws

env > env


ename="$( basename $exe )"
ofile="${ename}.out"
efile="${ename}.err"


export OMPI_MCA_btl_openib_allow_ib=1
mpirun -np 3 $exe 1>$ofile 2>$efile

cat $ofile $efile 

echo "send and receive"
for rank in 1 2; do
    otest "rank 0: sending 10 bytes to rank ${rank}"
    otest "rank ${rank}: received 10 bytes from 0, the message is hello!"
done

echo "distribuition and mpi_gather_allv"
for rank in 0 1 2; do
    times=33
    (( rank < 1 )) && times=34

    pattern="$( repeat $rank $times )"
    otest "local for rank ${rank}: ${pattern}"

    pattern="$( repeat 0 34 )$( repeat 1 33 )$( repeat 2 33 )"
    otest "global for rank ${rank}: ${pattern}"
done 


echo "mpi_scatterv" 
pattern="$( repeat 1 34 )$( repeat 2 33 )$( repeat 3 33 )"
otest "global for rank 0: ${pattern}"

for rank in 0 1 2; do
    times=33
    (( rank < 1 )) && times=34

    pattern="$( repeat $(( rank + 1 )) $times )"
    otest "local for rank ${rank}: ${pattern}"
done
