#ifndef __MYMPI_HPP_GUARD__
#define __MYMPI_HPP_GUARD__


extern "C" {
#include "mympi.h"
}

#include "print.hpp"


#define self (*this)

namespace mpi {

template <typename T>
class Distribution; 

class Communicator {
    int mranks{ 0 }; 
    int mrank{ 0 }; 

    public: 
    MpiState cstate; 

    Communicator() {
        mpi_initialize( &self.cstate ); 

        self.mranks = mpi_ranks( self.cstate );
        self.mrank = mpi_rank( self.cstate ); 
        $log( "MPI initialized, rank", self.rank(), "of", self.ranks() ); 
    }
    ~Communicator() {
        mpi_finalize( self.cstate ); 
        $log( "MPI finalized, rank", self.rank(), "of", self.ranks() ); 
    }

    int ranks() const noexcept { return self.mranks; } 
    int rank() const noexcept { return self.mrank; } 
    bool master() const noexcept { return (self.rank() == 0); }
    

    template <typename T>
    void send(const T* data, int count, int to) const {
        $log( self.rank(), "sent", count * sizeof(T), "bytes to", to ); 
        if ( count > 0 and to != self.rank() ) {
            mpi_send( self.cstate, (const void*)data, sizeof(T) * count, to ); 
        }
    }

    template <typename T>
    void receive(T* data, int count, int from) const {
        $log( self.rank(), "got", count * sizeof(T), "bytes from", from ); 
        if ( count > 0 and from != self.rank() ) {
            mpi_recv( self.cstate, (void*)data, sizeof(T) * count, from ); 
        }
    }

    template <typename T>
    void scatter(T* dst, const T* src, const Distribution<T>& distr) const; 
    
    template <typename T>
    void gather(T* dst, const T* src, const Distribution<T>& distr) const; 
    
    template <typename T>
    void gather_all(T* dst, const T* src, const Distribution<T>& distr) const; 
}; 


template <typename T> 
class Distribution {
    friend class Communicator; 

    const Communicator* mcomm{ nullptr }; 
    const unsigned mtotal{ 0 }; 
    MpiDistribution cdistr; 
    
    public: 
    Distribution(const Communicator* comm, unsigned total) 
        : mcomm{ comm }, mtotal{ total }
    {
        mpi_distribution_init( 
            &self.cdistr, mcomm->cstate, total, sizeof(T)
        ); 
    }
    ~Distribution() {
        mpi_distribution_free( self.cdistr ); 
    }

    const Communicator& comm() const { return *(self.mcomm); }
    
    unsigned total() const noexcept { return self.mtotal;  }

    unsigned count(int rank) const {
        return mpi_distribution_bcount( self.cdistr, rank ) / sizeof(T); 
    }
    unsigned count() const noexcept {
        return self.count( self.rank() ); 
    } 
    unsigned offset(int rank) const {
        return mpi_distribution_boffset( self.cdistr, rank ) / sizeof(T); 
    }
    unsigned offset() const noexcept {
        return self.offset( self.rank() ); 
    } 
    
    unsigned rank() const noexcept {
        return self.comm().rank(); 
    }
    unsigned ranks() const noexcept {
        return self.comm().ranks(); 
    }

    void scale(int factor) {
        mpi_distribution_scale( self.cdistr, factor ); 
    }


    void scatter(T* dst, const T* src) const {
        self.comm().scatter( dst, src, self ); 
    }

    void gather(T* dst, const T* src) const {
        self.comm().gather( dst, src, self ); 
    }
    void gather_all(T* dst, const T* src) const {
        self.comm().gather_all( dst, src, self ); 
    }
}; // class Distribution 

template <typename T>
std::ostream& operator << (std::ostream& os, const Distribution<T>& d) {
    os << "Distribution: "; 
    for (unsigned rank{0}; rank < d.ranks(); ++rank) {
        os << d.count( rank ) << "|" << d.offset( rank ) << " "; 
    }
    return os; 
}


template <typename T>
void Communicator::scatter(T* dst, const T* src, const Distribution<T>& distr) const {
    mpi_scatterv( 
        distr.cdistr, (void*)dst, (const void*)src
    ); 
}

template <typename T>
void Communicator::gather(T* dst, const T* src, const Distribution<T>& distr) const {
    $log( "gatherall", distr ); 
    mpi_gatherv( 
        distr.cdistr, (void*)dst, (const void*)src
    ); 
} 
template <typename T>
void Communicator::gather_all(T* dst, const T* src, const Distribution<T>& distr) const {
    $log( "gather_all", distr ); 
    mpi_gather_allv( 
        distr.cdistr, (void*)dst, (const void*)src
    ); 
} 


class Timer {
    MpiTimer ctimer; 

    public: 
    Timer() {
        mpi_timer_init( &self.ctimer );  
    }
    ~Timer() {
        mpi_timer_free( self.ctimer ); 
    } 

    void start() noexcept {
        mpi_timer_start( self.ctimer ); 
    }

    double stop() noexcept {
        return mpi_timer_stop( self.ctimer ); 
    }
        
    double seconds() const noexcept {
        return mpi_timer_seconds( self.ctimer ); 
    }; 
    double total() const noexcept {
        return self.seconds(); 
    }
}; // class Timer 
} // namespace mpi 
#undef self
#endif // __MYMPI_HPP_GUARD__
