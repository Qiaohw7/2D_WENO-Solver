// #ifndef Timestep_HPP
// #define Timestep_HPP

#include <iostream>
#include <stddef.h>
#include <algorithm>
#include <Kokkos_Core.hpp>
#include "flcl-cxx.hpp"
#include "Controlpara.hpp"

// Define the data types used in the code
using exec_space = Kokkos::DefaultExecutionSpace;
using view_type = flcl::view_r64_3d_t;
using view_type_r1d = flcl ::view_r64_1d_t;



void Timestep( view_type &U, double &dt){

    // int Jx = c_Nx/4;
    // int Jy = c_Ny/5;
    double dx = c_Lx/c_Nx;
    double dy = c_Ly/c_Ny;
    double CFL = c_CFL;
    double Gamma = c_Gamma;
    double maxvel;

    maxvel = 1e-8;

    #if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
    Kokkos::parallel_reduce( "Timestep", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {c_Nx+7, c_Ny+7}),  KOKKOS_LAMBDA(const size_t i, const size_t j, double& thread_maxvel)
    {
        
            double rhol = U(i,j,0);
            double ul   = U(i,j,1)/rhol;
            double vl   = U(i,j,2)/rhol;
            double pl   = (Gamma-1)*(U(i,j,3)-0.5*rhol*(ul*ul+vl*vl));
            double vel  = sqrt(fabs(Gamma*pl/rhol))+ sqrt(fabs(ul*ul+vl*vl));
            
                if(vel<0){vel=1.e-8;}

            if(vel > thread_maxvel){thread_maxvel = vel;} 
        
    }, Kokkos::Max<double>(maxvel));  
    #endif

    Kokkos::fence();
    // std::cout << "cfl = " << CFL << "dx" << dx << "dy" << dy << std::endl;
    
    dt = CFL * std::min(dx,dy) / maxvel;
     
}

// #endif