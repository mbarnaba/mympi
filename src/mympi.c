#include "mympi.h"

#include <mpi.h>
#include <string.h>


#define TAG (0)
#define COMMUNICATOR MPI_COMM_WORLD
#define MASTER_RANK (0)


struct pMpiState {
    int rank; 
    int ranks; 
    MPI_Comm comm; 
}; 
int mpi_initialize(MpiState* statep) {
    MpiState state = malloc( sizeof(struct pMpiState) ); 
    *statep = state; 

    state->comm = COMMUNICATOR; 

    int ret = MPI_Init( NULL, NULL );
    ret = MPI_Comm_size( state->comm, &state->ranks ); 
    ret = MPI_Comm_rank( state->comm, &state->rank ); 
    return ret; 
}
int mpi_finalize(MpiState state) {  
    free( state );
    MPI_Finalize(); 
    return 0; 
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
    unsigned* bcounts; 
    unsigned* boffsets; 
}; 
void mpi_distribution_init(MpiDistribution* distrp, const MpiState state, unsigned total, unsigned belm) {
    struct pMpiDistribution tmp = { .state = state }; 

    MpiDistribution distr = malloc( sizeof(struct pMpiDistribution) );  
    *distrp = distr; 
    memcpy( distr, &tmp, sizeof(struct pMpiDistribution) ); 

    const unsigned ranks = state->ranks; 
    const unsigned perrank = total / ranks; 
    const unsigned remainder = total - (perrank * ranks); 

    distr->bcounts = malloc( 2 * sizeof(unsigned) * ranks ); 
    distr->boffsets = distr->bcounts + (sizeof(unsigned) * ranks); 
    
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


int mpi_scatterv(const MpiDistribution distr, void* dst, const void* src) {
    const unsigned ranks = mpi_ranks( distr->state ); 
    const unsigned total = distr->bcounts[ ranks - 1 ] + distr->boffsets[ ranks - 1 ];  

    // https://www.mpich.org/static/docs/v3.1/www3/MPI_Scatterv.html
    return MPI_Scatterv(
        src, (const int*)distr->bcounts, (const int*)distr->boffsets, MPI_CHAR, dst, total, MPI_CHAR, MASTER_RANK, distr->state->comm 
    ); 
}

int mpi_gatherv(const MpiDistribution distr, void* dst, const void* src) {
    const unsigned sendcount = distr->bcounts[ mpi_rank(distr->state) ];  

    // https://www.mpich.org/static/docs/v3.1/www3/MPI_Gatherv.html
    return MPI_Gatherv(
        src, sendcount, MPI_CHAR, dst, (const int*)distr->bcounts, (const int*)distr->boffsets, MPI_CHAR, MASTER_RANK, distr->state->comm
    ); 
}

int mpi_gather_allv(const MpiDistribution distr, void* dst, const void* src) {
    const unsigned sendcount = mpi_distribution_bcount( 
        distr, mpi_rank(distr->state) 
    );  
    
    // https://www.mpich.org/static/docs/v3.2/www3/MPI_Allgatherv.html
    return MPI_Allgatherv(
        src, sendcount, MPI_CHAR, dst, (const int*)distr->bcounts, (const int*)distr->boffsets, MPI_CHAR, distr->state->comm
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
