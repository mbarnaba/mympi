#ifndef __MYMPI_H_GUARD__
#define __MYMPI_H_GUARD__


#include <stdlib.h>
#include <stdio.h>


struct pMpiState; 
#define MpiState struct pMpiState* 

int mpi_initialize(MpiState* state); 
int mpi_finalize(MpiState state); 

int mpi_rank(const MpiState state); 
int mpi_ranks(const MpiState state); 


void mpi_send(const MpiState state, const void* data, int bytes, int to); 
void mpi_recv(const MpiState state, void* data, int bytes, int from); 


struct pMpiDistribution; 
#define MpiDistribution struct pMpiDistribution* 

void mpi_distribution_init(MpiDistribution* distrp, const MpiState state, unsigned total, unsigned belm); 
void mpi_distribution_free(MpiDistribution distr); 
void mpi_distribution_print(const MpiDistribution distr); 

unsigned mpi_distribution_bcount(const MpiDistribution distr, unsigned rank); 
unsigned mpi_distribution_boffset(const MpiDistribution distr, unsigned rank); 
unsigned mpi_distribution_btotal(const MpiDistribution distr); 

void mpi_distribution_scale(MpiDistribution distr, int factor); 


int mpi_scatterv(const MpiDistribution distr, unsigned root, const void* src, void* dst); 
int mpi_gatherv(const MpiDistribution distr, unsigned root, const void* src, void* dst); 
int mpi_gather_allv(const MpiDistribution distr, const void* src, void* dst); 


struct pMpiTimer;  
#define MpiTimer struct pMpiTimer*

void mpi_timer_init(MpiTimer* timer); 
void mpi_timer_free(MpiTimer timer); 

void mpi_timer_start(MpiTimer timer); 
double mpi_timer_stop(MpiTimer timer); 
double mpi_timer_seconds(const MpiTimer timer); 
double mpi_timer_total(const MpiTimer timer); 

#endif // __MYMPI_H_GUARD__
