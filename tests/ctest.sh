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

if which module; then
    module purge
    module load gcc cmake 
    module load openmpi/4.0.1/gcc/8.2.0-2wc6vws
    env > env
fi


ename="$( basename $exe )"
ofile="${ename}.out"
efile="${ename}.err"


export OMPI_MCA_btl_openib_allow_ib=1
#--oversubscribe \
mpirun \
    -np 3 \
    $exe 1>$ofile 2>$efile

cat $ofile $efile 

echo "send and receive"
for rank in 1 2; do
    otest "rank 0: sending 10 bytes to rank ${rank}"
    otest "rank ${rank}: received 10 bytes from 0, the message is hello!"
done


echo "mpi_gatherv"
for rank in 0 1 2; do
    pattern="distribuition (rank ${rank}): (136 0) (132 136) (132 268)"
    otest "$pattern"

    count=33
    (( rank < 1 )) && count=34 
    pattern="rank ${rank}: bcount = $(( count * 4 )), count = ${count}"
    otest "$pattern"

    if (( rank == 0 )); then
        pattern=''
        for rank2 in 0 1 2; do
            times=33
            (( rank2 < 1 )) && times=34
            pattern="${pattern}$( repeat $rank2 $times )"
        done

        pattern="global for rank 0: ${pattern}"
        otest "$pattern"
    fi
done 


echo "mpi_scatterv"
for rank in 0 1 2; do
    times=33
    (( rank < 1 )) && times=34
    
    pattern="$( repeat $(( rank + 1 )) $times )"
    pattern="local for rank ${rank}: ${pattern}"
    otest "$pattern"
done 


echo "mpi_gather_allv"
for rank in 0 1 2; do
    times=33
    (( rank < 1 )) && times=34
    
    pattern=''
    for rank2 in 0 1 2; do
        times2=33
        (( rank2 < 1 )) && times2=34
        pattern="${pattern}$( repeat $(( rank2 + 2 )) $times2 )"
    done

    pattern="global for rank ${rank}: ${pattern}"
    otest "$pattern"
done 
