#ifndef __MYMPI_HPP_GUARD__
#define __MYMPI_HPP_GUARD__


extern "C" {
#include "mympi.h"
}

#include "print.hpp"


#define self (*this)

namespace mympi {

class Handle {
    unsigned mrank{ 0 };
    unsigned mranks{ 0 }; 

    public:
    MpiState cstate; 

    Handle() {
        mpi_initialize( &self.cstate ); 
        self.mrank = mpi_rank( self.cstate ); 
        self.mranks = mpi_ranks( self.cstate ); 
        $log( 
            "Handle for process", self.rank() + 1, "(rank ", 
            self.rank(), ") of", self.ranks(), "successfuly initialized"
        ); 
    }
    ~Handle() {
        mpi_finalize( self.cstate ); 
        $log( 
            "Handle for process", self.rank() + 1, "(rank ", 
            self.rank(), ") of", self.ranks(), "successfuly finalized"
        ); 
    }

    unsigned ranks() const noexcept {
        return self.mranks; 
    }
    unsigned rank() const noexcept { 
        return self.mrank; 
    }

    bool master(unsigned rank) const noexcept {
        return (rank == 0); 
    }
    bool master() const noexcept {
        return self.master( self.rank() ); 
    }


    template <typename T>
    void send(const T* src, unsigned count, unsigned to) const {
        mpi_send( 
            self.cstate, 
            static_cast<const void*>(src), 
            count * sizeof(T), 
            to 
        ); 
    }

    template <typename T>
    void receive(T* dst, unsigned count, unsigned from) const {
        mpi_recv( 
            self.cstate, 
            static_cast<void*>(dst), 
            count * sizeof(T), 
            from 
        ); 
    }

    
    template <typename T>
    void bcast(T* buffer, unsigned count, unsigned root=0) const {
        mpi_bcast(
            self.cstate, 
            static_cast<void*>(buffer), 
            count * sizeof(T), 
            root
        ); 
    }
    

    void sum_all(const double* src, double* dst, unsigned count) const {
        mpi_dsum_all( 
            self.cstate, 
            count, 
            src, 
            dst
        );
    }
    void sum_all(const int* src, int* dst, unsigned count) const {
        mpi_isum_all( 
            self.cstate, 
            count, 
            src, 
            dst
        );
    }
}; 


template <typename T>
class Distribution {
    MpiDistribution cdistr; 
    const unsigned mtotal; 
    const Handle* const mhandle; 
    int mfactor{ 1 }; 

    public: 
    Distribution(const Handle* handle, unsigned total) 
        : mtotal{ total }, mhandle{ handle }
    {
        mpi_distribution_init( 
            &self.cdistr, 
            self.handle().cstate, 
            total, 
            sizeof(T) 
        ); 
        $log(
            "Distribution with total", self.total(), "and factor", self.factor(), 
            "successfuly initialized"
        ); 
    }
    ~Distribution() noexcept {
        mpi_distribution_free( self.cdistr ); 
        $log(
            "Distribution with total", self.total(), "and factor", self.factor(), 
            "successfuly finalized"
        ); 
    }


    unsigned total() const noexcept { 
        return self.mtotal; 
    }

    const Handle& handle() const noexcept {
        return *(self.mhandle); 
    }

    int factor() const noexcept { 
        return self.mfactor;
    } 
    int factor(int factor) noexcept {
        self.scale( factor );  
        return self.factor(); 
    }
    void scale(int factor) noexcept {
        if ( factor > 1 or factor < -1 ) { 
            mpi_distribution_scale( self.cdistr, factor ); 
            self.mfactor = factor;  
        }
    }

    
    unsigned rank() const noexcept {
        return self.handle().rank(); 
    }
    unsigned ranks() const noexcept {
        return self.handle().ranks(); 
    }

    unsigned count(unsigned rank) const {
        return mpi_distribution_bcount( 
            self.cdistr, rank
        ) / sizeof(T); 
    } 
    unsigned count() const noexcept {
        return self.count( self.rank() ); 
    }

    unsigned offset(unsigned rank) const {
        return mpi_distribution_boffset(
            self.cdistr, rank
        ) / sizeof(T); 
    }
    unsigned offset() const noexcept {
        return self.offset( self.rank() ); 
    }

    
    void scatter(const T* src, T* dst, unsigned root=0) const {
        mpi_scatterv( 
            self.cdistr, 
            root, 
            static_cast<const void*>( src ), 
            static_cast<void*>( dst ) 
        ); 
    }

    void gather(const T* src, T* dst, unsigned root=0) const {
        mpi_gatherv(
            self.cdistr, 
            root, 
            static_cast<const void*>(src), 
            static_cast<void*>(dst)
        ); 
    }

    void gather_all(const T* src, T* dst) const {
        mpi_gather_allv(
            self.cdistr, 
            static_cast<const void*>(src), 
            static_cast<void*>(dst)
        ); 
    }    
};  

template <typename T>
std::ostream& operator << (std::ostream& os, const Distribution<T>& distr) {
    for (unsigned rank{ 0 }; rank < distr.ranks(); ++rank) {
        os << "(" << distr.count(rank) << ", " << distr.offset(rank) << ") ";
    }
    return os; 
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

    void start() {
        mpi_timer_start( self.ctimer ); 
    }
    double stop() {
        return mpi_timer_stop( self.ctimer ); 
    }

    double seconds() const {
        return mpi_timer_seconds( self.ctimer ); 
    }
    double total() const {
        return mpi_timer_total( self.ctimer ); 
    }
}; 
} // namespace mympi
#undef self
#endif // __MYMPI_HPP_GUARD__
