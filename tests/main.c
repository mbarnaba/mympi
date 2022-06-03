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

    // initialize local
    for (unsigned idx = 0; idx < count; idx++) {
        local[ idx ] = rank; 
    }
    printf( "local for rank %d: ", rank ); 
    vector_print( local, count ); 
    
    const int master = 0; 
    // mpi_gatherv will populate global for the master 
    mpi_distribution_print( distr ); 
    mpi_gatherv( distr, master, local, global ); 
    // let's check, for the master only (for other ranks we would print garbage)
    if (rank == master) {
        printf( "global for rank %d: ", rank ); 
        vector_print( global, total ); 

        // master will modify global before scattering back to all ranks 
        for (unsigned idx = 0; idx < total; idx++) {
            global[ idx ] += 1; 
        }
    }

    mpi_scatterv( distr, master, global, local ); 
    // now local should be modified (+1), let's check
    printf( "local for rank %d: ", rank ); 
    vector_print( local, count ); 

    // all ranks will now modify local and call mpi_gather_allv
    for (unsigned idx = 0; idx < count; idx++) {
        local[ idx ] += 1; 
    } 
    // reading from local and writing to global
    mpi_gather_allv( distr, local, global ); 
    // now global should be modified (+1), let's check
    printf( "global for rank %d: ", rank ); 
    vector_print( global, total ); 
    
    mpi_distribution_free( distr ); 
}


void test_reduce(const MpiState state) {
    const unsigned count = 128; 

    double src[ count ]; 
    double dst[ count ]; 
    
    for (unsigned i=0; i < count; i++) {
        src[ i ] = mpi_rank( state ); 
    }
    
    mpi_dsum_all( state, count, src, dst );     

    const double expected = mpi_ranks( state ); 

    for (unsigned i=0; i < count; i++) {
        if ( dst[ i ] != expected ) {
            printf( "did not found expected value after mpi_dsum_all!" );
            exit( 1 ); 
        }
    }
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


    test_reduce( state ); 

    test_distribution( state ); 

    mpi_finalize( state ); 
} 
