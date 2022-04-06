#include "mympi.h"


#define TYPE int

static void vector_print(const int* data, unsigned count) {
    for (unsigned idx = 0; idx < count; idx++) {
        printf( "%d", data[ idx ] ); 
    }
    printf( "\n" ); 
}

static void test_distribution(const MpiState state) {
    const int rank = mpi_rank( state ); 

    const unsigned total = 100; 
    printf( "total = %d \n", total ); 

    MpiDistribution distr; 
    mpi_distribution_init( &distr, state, total, sizeof(TYPE) );

    const unsigned bcount = mpi_distribution_bcount( distr, rank ); 
    const unsigned count = bcount / sizeof(TYPE); 
    const unsigned offset = mpi_distribution_boffset( distr, rank ) / sizeof(TYPE); 
    printf( "rank %d: bcount = %d, count = %d \n", rank, bcount, count ); 
    printf( "rank %d: offset = %d \n", rank, offset ); 

    const unsigned btotal = mpi_distribution_btotal( distr ); 
    printf( "btotal = %d\n", btotal ); 

    TYPE* global = malloc( btotal ); 
    TYPE* local = malloc( bcount ); 
    for (unsigned idx = 0; idx < count; idx++) {
        local[ idx ] = rank; 
    }
    
    // mpi_gather_allv
    mpi_gather_allv( distr, global, local ); 
    printf( "global for rank %d: ", rank ); 
    vector_print( global, total ); 
    printf( "local for rank %d: ", rank ); 
    vector_print( local, count ); 

    
    if (rank == 0) {
        for (unsigned idx = 0; idx < total; idx++) {
            global[ idx ] += 1; 
        }
        printf( "global for rank %d: ", rank ); 
        vector_print( global, total ); 
    } 

    // mpi_scatterv
    mpi_scatterv( distr, local, global ); 
    printf( "local for rank %d: ", rank ); 
    vector_print( local, count ); 


    // mpi_gatherv
    for (unsigned idx = 0; idx < count; idx++) {
        local[ idx ] += 1; 
    }
    printf( "local for rank %d: ", rank ); 
    vector_print( local, count ); 

    mpi_gatherv( distr, global, local );  
    if (rank == 0) {
        printf( "global for rank %d: ", rank ); 
        vector_print( global, total ); 
    }

    mpi_distribution_free( distr ); 
}


int main(void) {
    MpiState state;
    mpi_initialize( &state );
    
    const int rank = mpi_rank( state ); 
    const int ranks = mpi_ranks( state );  

    if (rank == 0) {
        printf( "process %d of %d (rank %d) \n", rank + 1, ranks, rank ); 
    
        const char msg[ 128 ] = "hello!"; 
        for (unsigned to = 1; to < 3; to++) {
            printf( "rank %d: sending 10 bytes to rank %d\n", rank, to ); 
            mpi_send( state, (const void*) msg, 10, to ); 
        }
    } else {
        char msg[ 128 ]; 
        const int from = 0; 
        mpi_recv( state, (void*) msg, 10, from ); 
        printf( "rank %d: received 10 bytes from %d, the message is %s\n", rank, from, msg ); 
    }

    test_distribution( state ); 

    mpi_finalize( state ); 
} 
