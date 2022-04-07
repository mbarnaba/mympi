#include "mympi.h"

#include <mpi.h>
#include <string.h>


#define TAG (0)
#define MASTER_RANK (0)


struct pMpiState {
    int rank; 
    int ranks; 
    MPI_Comm comm; 
}; 
int mpi_initialize(MpiState* statep) {
    MpiState state = malloc( sizeof(struct pMpiState) ); 
    *statep = state; 
    
    int ret = MPI_Init( NULL, NULL );

    state->comm = MPI_COMM_WORLD; 

    ret = MPI_Comm_size( state->comm, &state->ranks ); 
    ret = MPI_Comm_rank( state->comm, &state->rank ); 
    return ret; 
}
int mpi_finalize(MpiState state) {  
    free( state );
    return MPI_Finalize(); 
}


int mpi_rank(const MpiState state) {
    return state->rank; 
}
int mpi_ranks(const MpiState state) {
    return state->ranks; 
}


void mpi_send(const MpiState state, const void* data, int bytes, int to) {
    MPI_Send(
        data, bytes, MPI_CHAR, to, TAG, state->comm
    ); 
}
void mpi_recv(const MpiState state, void* data, int bytes, int from) {
    MPI_Recv(
        data, bytes, MPI_CHAR, from, TAG, state->comm, NULL
    ); 
}


struct pMpiDistribution {
    const MpiState state; 
    int* bcounts; 
    int* boffsets; 
}; 
void mpi_distribution_init(MpiDistribution* distrp, const MpiState state, unsigned total, unsigned belm) {
    const struct pMpiDistribution tmp = { .state = state }; 

    MpiDistribution distr = malloc( sizeof(struct pMpiDistribution) );  
    *distrp = distr; 
    memcpy( distr, &tmp, sizeof(struct pMpiDistribution) ); 

    const unsigned ranks = mpi_ranks( state );
    const unsigned perrank = total / ranks; 
    const unsigned remainder = total - (perrank * ranks); 

    distr->bcounts = malloc( 2 * sizeof(unsigned) * ranks ); 
    distr->boffsets = distr->bcounts + ranks; 
    
    for (unsigned idx = 0; idx < ranks; idx++) {
        distr->bcounts[ idx ] = perrank * belm; 
        if (idx < remainder) {
            distr->bcounts[ idx ] += belm; 
        }

        if (idx > 0) {
            distr->boffsets[ idx ] = distr->boffsets[ idx - 1 ] + distr->bcounts[ idx - 1 ]; 
        } else {
            distr->boffsets[ 0 ] = 0; 
        }
    }
}
void mpi_distribution_free(MpiDistribution distr) {
    free( distr->bcounts ); 
    free( distr ); 
}

void mpi_distribution_print(const MpiDistribution distr) {
    const unsigned ranks = mpi_ranks( distr->state );
    printf( "distribuition (rank %d): ", mpi_rank( distr->state ) ); 
    for (unsigned rank = 0; rank < ranks; rank++) {
        printf(
            "(%d %d) ", 
            mpi_distribution_bcount( distr, rank ), 
            mpi_distribution_boffset( distr, rank )
        ); 
    }
    printf( "\n" ); 
}


unsigned mpi_distribution_bcount(const MpiDistribution distr, unsigned rank) {
    return distr->bcounts[ rank ]; 
} 
unsigned mpi_distribution_boffset(const MpiDistribution distr, unsigned rank) {
    return distr->boffsets[ rank ]; 
} 
unsigned mpi_distribution_btotal(const MpiDistribution distr) {
    const unsigned last = mpi_ranks( distr->state ) - 1; 
    return 
        mpi_distribution_boffset( distr, last ) 
        + mpi_distribution_bcount( distr, last ); 
} 


static inline void mpi_distribution_mul(MpiDistribution distr, unsigned factor) {
    const unsigned ranks = mpi_ranks( distr->state ); 
    for (unsigned idx = 0; idx < ranks; idx++) {
        distr->bcounts[ idx ] *= factor;   
        distr->boffsets[ idx ] *= factor;   
    }
}
static inline void mpi_distribution_div(MpiDistribution distr, unsigned factor) {
    const unsigned ranks = mpi_ranks( distr->state ); 
    for (unsigned idx = 0; idx < ranks; idx++) {
        distr->bcounts[ idx ] /= factor;   
        distr->boffsets[ idx ] /= factor;   
    }
}
void mpi_distribution_scale(MpiDistribution distr, int factor) {
    if (factor < 0) {
        mpi_distribution_div( distr, (-factor) ); 
    } else {
        mpi_distribution_mul( distr, factor );
    }
}


int mpi_scatterv(const MpiDistribution distr, unsigned root, const void* src, void* dst) {
    const int recvcount = mpi_distribution_btotal( distr ); 
    //const int rank = mpi_rank( distr->state ); 
    //printf( "mpi_scatterv rank %d recvcount %d from %d \n", rank, recvcount, root ); 
    //mpi_distribution_print( distr ); 

    /*
    int MPI_Scatterv(
        const void *sendbuf, 
        const int sendcounts[], const int displs[],
        MPI_Datatype sendtype, 
        void *recvbuf, int recvcount,
        MPI_Datatype recvtype, 
        int root, MPI_Comm comm
    )
    */
    return MPI_Scatterv(
        src, 
        distr->bcounts, 
        distr->boffsets, 
        MPI_CHAR, 
        dst, 
        recvcount, 
        MPI_CHAR, 
        root, 
        distr->state->comm
    ); 
}
int mpi_scatter(const MpiState state, unsigned total, unsigned root, const void *src, void* dst) {
    /* 
    https://www.open-mpi.org/doc/v4.1/man3/MPI_Scatter.3.php
          
    int MPI_Scatter(
        const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, int recvcount, MPI_Datatype recvtype, 
        int root, MPI_Comm comm
    )
    */

    const unsigned ranks = mpi_ranks( state ); 
    const unsigned recvcount = total / ranks; 
    return MPI_Scatter(
        src, 
        total,
        MPI_CHAR, 
        dst, 
        recvcount,
        MPI_CHAR, 
        root, 
        state->comm
    ); 
}


int mpi_gatherv(const MpiDistribution distr, unsigned root, const void* src, void* dst) {
    const unsigned rank = mpi_rank( distr->state ); 
    const int sendcount = mpi_distribution_bcount( distr, rank ); 
    //printf( "mpi_gatherv rank %d sendcount %d to %d \n", rank, sendcount, root ); 
    //mpi_distribution_print( distr ); 

    /* 
    https://www.open-mpi.org/doc/v4.1/man3/MPI_Gatherv.3.php 
          
    int MPI_Gatherv(
        const void *sendbuf, int sendcount, MPI_Datatype sendtype,
        void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype
        recvtype,
        int root, MPI_Comm comm
    )
    */
    return MPI_Gatherv(
        src, 
        sendcount, 
        MPI_CHAR, 
        dst, 
        distr->bcounts, 
        distr->boffsets, 
        MPI_CHAR, 
        root, 
        distr->state->comm
    ); 
}

int mpi_gather_allv(const MpiDistribution distr, const void* src, void* dst) {
    const unsigned rank = mpi_rank( distr->state ); 
    const int sendcount = mpi_distribution_bcount( distr, rank ); 
    //printf( "mpi_gather_allv rank %d sendcount %d \n", rank, sendcount ); 
    //mpi_distribution_print( distr ); 

    /*
    https://www.open-mpi.org/doc/v4.1/man3/MPI_Allgatherv.3.php

    int MPI_Allgatherv(
        const void *sendbuf, int sendcount, MPI_Datatype sendtype, 
        void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, 
        MPI_Comm comm
    );
    */
    return MPI_Allgatherv(
        src, 
        sendcount, 
        MPI_CHAR, 
        dst, 
        distr->bcounts, 
        distr->boffsets, 
        MPI_CHAR, 
        distr->state->comm
    ); 
} 


struct pMpiTimer {
    double seconds; 
}; 
void mpi_timer_init(MpiTimer* timerp) {
    MpiTimer timer = malloc( sizeof(struct pMpiTimer) );
    *timerp = timer; 

    timer->seconds = 0; 
}
void mpi_timer_free(MpiTimer timer) {
    free( timer ); 
}

void mpi_timer_start(MpiTimer timer) {
    timer->seconds -= MPI_Wtime();  
}
double mpi_timer_stop(MpiTimer timer) {
    timer->seconds += MPI_Wtime();  
    return mpi_timer_seconds( timer ); 
}
double mpi_timer_seconds(const MpiTimer timer) {
    return timer->seconds; 
}
