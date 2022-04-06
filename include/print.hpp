#ifndef __PRINT_HPP_GUARD__
#define __PRINT_HPP_GUARD__


#include <iostream>


template <typename S, typename D, typename V> 
void $sdprint(S& stream, const D&, const V& value) {
    stream << value; 
    stream << std::endl; 
} 
template <typename S, typename D, typename F, typename... O>
void $sdprint(S& stream, const D& delimiter, const F& first, const O&... others) {
    stream << first; 
    stream << delimiter; 
    $sdprint( stream, delimiter, others... ); 
} 

template <typename D, typename F, typename... O>
void $dprint(const D& delimiter, const F& first, const O&... others) {
    $sdprint( std::cout, delimiter, first, others... ); 
}
template <typename F, typename... O>
void $print(const F& first, const O&... others) {
    $dprint( ' ', first, others... ); 
}


#ifdef NDEBUG
template <typename F, typename... O>
void $log(const F&, const O&...) {
    //$print( ' ', first, others... ); 
}
#else
template <typename F, typename... O>
void $log(const F& first, const O&... others) {
    $print( ' ', first, others... ); 
}
#endif
#endif // __PRINT_HPP_GUARD__
