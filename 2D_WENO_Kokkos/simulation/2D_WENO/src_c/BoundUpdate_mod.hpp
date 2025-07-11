#ifndef BoundUpdate_HPP
#define BoundUpdate_HPP


#include <iostream>
#include <stddef.h>
#include <Kokkos_Core.hpp>
#include "flcl-cxx.hpp"
#include "Controlpara.hpp"

// Define the data types used in the code
using exec_space = Kokkos::DefaultExecutionSpace;
using view_type = flcl::view_r64_3d_t;

void BoundUpdate(view_type &U) {
    
// - local value
    int Jx = c_Nx/4;
    int Jy = c_Ny/5;
    double T_inf = c_P_inf / (c_Rho_inf * c_R);
    double U_inf = c_Mach_inf * sqrt(c_Gamma * c_R * T_inf);
    double V_inf = 0.0;
 
//     //Kokkos::parallel_for( "inflow", Kokkos::RangePolicy(3,c_Ny+4), KOKKOS_LAMBDA(const size_t j)
//     Kokkos::parallel_for( "inflow",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({0, 3}, {4, c_Ny+4}),
//     KOKKOS_LAMBDA(const size_t i, const size_t j)
//     {
//             U(i,j,0) = U(6-i,j,0);
//             U(i,j,1) = U(6-i,j,1);
//             U(i,j,2) = U(6-i,j,2);
//             U(i,j,3) = U(6-i,j,3);    
//     });
// Kokkos::fence();

//    //Kokkos::parallel_for( "lower-outflow", Kokkos::RangePolicy(0,c_Nx+7), KOKKOS_LAMBDA(const size_t i)
//     Kokkos::parallel_for( "lower-outflow",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({0, 0}, {c_Nx+7, 3}),
//     KOKKOS_LAMBDA(const size_t i, const size_t j)
//     {
//             U(i,j,0) = U(i,6-j,0);
//             U(i,j,1) = U(i,6-j,1);
//             U(i,j,2) = U(i,6-j,2);
//             U(i,j,3) = U(i,6-j,3); 
//     });
// Kokkos::fence();

//     //Kokkos::parallel_for( "upper-reflect-ghost", Kokkos::RangePolicy(0,Jx+7), KOKKOS_LAMBDA(const size_t i)
//     Kokkos::parallel_for( "upper-reflect-ghost",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({0, c_Ny+4}, {Jx+7, c_Ny+7}),
//     KOKKOS_LAMBDA(const size_t i, const size_t j)
//     {
//             U(i,j,0) = U(i,2*(c_Ny+3)-j,0);
//             U(i,j,1) = U(i,2*(c_Ny+3)-j,1);
//             U(i,j,2) = -U(i,2*(c_Ny+3)-j,2);
//             U(i,j,3) = U(i,2*(c_Ny+3)-j,3);
//     });
// Kokkos::fence();

//     Kokkos::parallel_for( "upper-reflect", Kokkos::RangePolicy(3,Jx+4), KOKKOS_LAMBDA(const size_t i)
//     {
//             U(i,c_Ny+3,0) = U(i,c_Ny+2,0);
//             U(i,c_Ny+3,1) = U(i,c_Ny+2,1);
//             U(i,c_Ny+3,2) = 0;
//             U(i,c_Ny+3,3) = U(i,c_Ny+2,3);    
//     });
// Kokkos::fence();

//     //Kokkos::parallel_for( "right-outflow-1", Kokkos::RangePolicy(0,Jy+4), KOKKOS_LAMBDA(const size_t j)
//     Kokkos::parallel_for( "right-outflow-1",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({c_Nx+4, 0}, {c_Nx+7, Jy+4}),
//     KOKKOS_LAMBDA(const size_t i, const size_t j)
//     {
//             U(i,j,0) = 0.001;
//             U(i,j,1) = 0.001;
//             U(i,j,2) = 0.001;
//             U(i,j,3) = 0.001;
//     });
// Kokkos::fence();

//     //Kokkos::parallel_for( "right-outflow-2", Kokkos::RangePolicy(Jy+4,c_Ny+7), KOKKOS_LAMBDA(const size_t j)
//     Kokkos::parallel_for( "right-outflow-2",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({c_Nx+4, Jy+4}, {c_Nx+7, c_Ny+7}),
//     KOKKOS_LAMBDA(const size_t i, const size_t j)
//     {
//             U(i,j,0) = (4*U(i-1,j,0)-U(i-2,j,0))/3.0;
//             U(i,j,1) = (4*U(i-1,j,1)-U(i-2,j,1))/3.0;
//             U(i,j,2) = (4*U(i-1,j,2)-U(i-2,j,2))/3.0;
//             U(i,j,3) = (4*U(i-1,j,3)-U(i-2,j,3))/3.0;
//     });

//     //Kokkos::parallel_for( "body-left-ghost", Kokkos::RangePolicy(3,Jy+4), KOKKOS_LAMBDA(const size_t j)
//     Kokkos::parallel_for( "body-left-ghost",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({Jx+4, 3}, {Jx+7, Jy+4}),
//     KOKKOS_LAMBDA(const size_t i, const size_t j)
//     {
//             U(i,j,0) = U(2*(Jx+3)-i,j,0);
//             U(i,j,1) = -U(2*(Jx+3)-i,j,1);
//             U(i,j,2) = U(2*(Jx+3)-i,j,2);
//             U(i,j,3) = U(2*(Jx+3)-i,j,3);
//     });
// Kokkos::fence();

//     Kokkos::parallel_for( "body-left", Kokkos::RangePolicy(3,Jy+4), KOKKOS_LAMBDA(const size_t j)
//     {
//             U(Jx+3,j,0) = U(Jx+2,j,0);
//             U(Jx+3,j,1) = 0;
//             U(Jx+3,j,2) = U(Jx+2,j,2);
//             U(Jx+3,j,3) = U(Jx+2,j,3);

//     });
// Kokkos::fence();

//     //Kokkos::parallel_for( "body-upper-ghost", Kokkos::RangePolicy(Jx+4,c_Nx+4), KOKKOS_LAMBDA(const size_t i)
//     Kokkos::parallel_for( "body-upper-ghost",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({Jx+4, Jy}, {c_Nx+4, Jy+3}),
//     KOKKOS_LAMBDA(const size_t i, const size_t j)
//     {
//             U(i,j,0) = U(i,2*(Jy+3)-j, 0);
//             U(i,j,1) = U(i,2*(Jy+3)-j, 1);
//             U(i,j,2) = -U(i,2*(Jy+3)-j, 2);
//             U(i,j,3) = U(i,2*(Jy+3)-j, 3);
//     });
// Kokkos::fence();

//     Kokkos::parallel_for( "body-upper", Kokkos::RangePolicy(Jx+4,c_Nx+4), KOKKOS_LAMBDA(const size_t i)
//     {
//         U(i,Jy+3,0) = (4*U(i,Jy+4,0) - U(i,Jy+5,0))/3.0;
//         U(i,Jy+3,1) = (4*U(i,Jy+4,1) - U(i,Jy+5,1))/3.0;
//         U(i,Jy+3,2) = 0;
//         U(i,Jy+3,3) = (4*U(i,Jy+4,3) - U(i,Jy+5,3))/3.0;
//     });
// Kokkos::fence();
   
// // -corner
//     U(Jx+4,Jy+2,0) = 0.5*( U(Jx+2,Jy+2,0)+U(Jx+4,Jy+4,0) );
//     U(Jx+4,Jy+2,1) = 0.5*( U(Jx+2,Jy+2,1)+U(Jx+4,Jy+4,1) );
//     U(Jx+4,Jy+2,2) = 0.5*( U(Jx+2,Jy+2,2)+U(Jx+4,Jy+4,2) );
//     U(Jx+4,Jy+2,3) = 0.5*( U(Jx+2,Jy+2,3)+U(Jx+4,Jy+4,3) );

//     U(Jx+4,Jy+1,0) = U(Jx+2,Jy+1,0);
//     U(Jx+4,Jy+1,1) = -U(Jx+2,Jy+1,1);
//     U(Jx+4,Jy+1,2) = U(Jx+2,Jy+1,2);
//     U(Jx+4,Jy+1,3) = U(Jx+2,Jy+1,3);

//     U(Jx+4,Jy,0) = U(Jx+2,Jy,0);
//     U(Jx+4,Jy,1) = -U(Jx+2,Jy,1);
//     U(Jx+4,Jy,2) = U(Jx+2,Jy,2);
//     U(Jx+4,Jy,3) = U(Jx+2,Jy,3);

//     U(Jx+5,Jy+2,0) = U(Jx+5,Jy+4,0);
//     U(Jx+5,Jy+2,1) = U(Jx+5,Jy+4,1);
//     U(Jx+5,Jy+2,2) = -U(Jx+5,Jy+4,2);
//     U(Jx+5,Jy+2,3) = U(Jx+5,Jy+4,3);

//     U(Jx+5,Jy+1,0) = 0.5*( U(Jx+5,Jy+5,0)+U(Jx+1,Jy+1,0) );
//     U(Jx+5,Jy+1,1) = 0.5*( U(Jx+5,Jy+5,1)+U(Jx+1,Jy+1,1) );
//     U(Jx+5,Jy+1,2) = 0.5*( U(Jx+5,Jy+5,2)+U(Jx+1,Jy+1,2) );
//     U(Jx+5,Jy+1,3) = 0.5*( U(Jx+5,Jy+5,3)+U(Jx+1,Jy+1,3) );

//     U(Jx+5,Jy,0) = U(Jx+1,Jy,0);
//     U(Jx+5,Jy,1) = -U(Jx+1,Jy,1);
//     U(Jx+5,Jy,2) = U(Jx+1,Jy,2);
//     U(Jx+5,Jy,3) = U(Jx+1,Jy,3);

//     U(Jx+6,Jy+2,0) = U(Jx+6,Jy+4,0);
//     U(Jx+6,Jy+2,1) = U(Jx+6,Jy+4,1);
//     U(Jx+6,Jy+2,2) = -U(Jx+6,Jy+4,2);
//     U(Jx+6,Jy+2,3) = U(Jx+6,Jy+4,3);

//     U(Jx+6,Jy+1,0) = U(Jx+6,Jy+5,0);
//     U(Jx+6,Jy+1,1) = U(Jx+6,Jy+5,1);
//     U(Jx+6,Jy+1,2) = -U(Jx+6,Jy+5,2);
//     U(Jx+6,Jy+1,3) = U(Jx+6,Jy+5,3);

//     U(Jx+6,Jy,0) = U(Jx+3,Jy+3,0);
//     U(Jx+6,Jy,1) = -U(Jx+3,Jy+3,1);
//     U(Jx+6,Jy,2) = -U(Jx+3,Jy+3,2);
//     U(Jx+6,Jy,3) = U(Jx+3,Jy+3,3);

// - inflow
    //Kokkos::parallel_for( "inflow", Kokkos::RangePolicy(3,c_Ny+4), KOKKOS_LAMBDA(const size_t j)
    Kokkos::parallel_for( "inflow",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({0, 3}, {4, c_Ny+4}),
    KOKKOS_LAMBDA(const size_t i, const size_t j)
    {
            U(i, j, 0) = c_Rho_inf;
            U(i, j, 1) = c_Rho_inf * U_inf;
            U(i, j, 2) = c_Rho_inf * V_inf;
            U(i, j, 3) = c_P_inf / (c_Gamma - 1) + 0.5 * c_Rho_inf * (U_inf * U_inf + V_inf * V_inf);
    });
Kokkos::fence();
   // Kokkos::parallel_for( "outflow", Kokkos::RangePolicy(Jy+4,c_Ny+4), KOKKOS_LAMBDA(const size_t j)
    Kokkos::parallel_for( "outflow",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({c_Nx+4, Jy+4}, {c_Nx+7, c_Ny+4}),
    KOKKOS_LAMBDA(const size_t i, const size_t j)
    {
            U(i,j,0) = U(c_Nx+3,j,0);
            U(i,j,1) = U(c_Nx+3,j,1);
            U(i,j,2) = U(c_Nx+3,j,2);
            U(i,j,3) = U(c_Nx+3,j,3);
    });
Kokkos::fence();
    Kokkos::parallel_for( "wall_upper_side-1", Kokkos::RangePolicy(3,c_Nx+4), KOKKOS_LAMBDA(const size_t i)
    {
        U(i,c_Ny+3,0) = (4.0*U(i,c_Ny+2,0)-U(i,c_Ny+1,0))/3.0;
        U(i,c_Ny+3,1) = (4.0*U(i,c_Ny+2,1)-U(i,c_Ny+1,1))/3.0;
        U(i,c_Ny+3,2) = 0;
        U(i,c_Ny+3,3) = (4.0*U(i,c_Ny+2,3)-U(i,c_Ny+1,3))/3.0;
    });
Kokkos::fence();
    //Kokkos::parallel_for( "wall_upper_side-2", Kokkos::RangePolicy(3,c_Nx+4), KOKKOS_LAMBDA(const size_t i)
    Kokkos::parallel_for( "wall_upper_side-2",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({3, c_Ny+4}, {c_Nx+4,c_Ny+7}),
    KOKKOS_LAMBDA(const size_t i, const size_t j)
    {
            U(i,j,0) =  U(i,2*(c_Ny+3)-j,0);  //U(i,j,1) =  U(i,2*c_Ny-j,1)
            U(i,j,1) =  U(i,2*(c_Ny+3)-j,1);
            U(i,j,2) =  0; //-U(i,2*(c_Ny+3)-j,2);;
            U(i,j,3) =  U(i,2*(c_Ny+3)-j,3);
    });
Kokkos::fence();
    Kokkos::parallel_for( "wall_lower_side-1-1", Kokkos::RangePolicy(3,Jx+4), KOKKOS_LAMBDA(const size_t i)
    {
        U(i,3,0) = (4.0*U(i,5,0)-U(i,6,0))/3.0;
        U(i,3,1) = (4.0*U(i,5,1)-U(i,6,1))/3.0;
        U(i,3,2) = 0;
        U(i,3,3) = (4.0*U(i,5,3)-U(i,6,3))/3.0;
    });
Kokkos::fence();
    //Kokkos::parallel_for( "wall_lower_side-1-2", Kokkos::RangePolicy(3,Jx+3), KOKKOS_LAMBDA(const size_t i)
    Kokkos::parallel_for( "wall_lower_side-1-2",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({3, 0}, {Jx+3,3}),
    KOKKOS_LAMBDA(const size_t i, const size_t j)
    {
            U(i,j,0) =  U(i,6-j,0);
            U(i,j,1) =  U(i,6-j,1);
            U(i,j,2) = 0;
            U(i,j,3) =  U(i,6-j,3);
    });
Kokkos::fence();
    Kokkos::parallel_for( "wall_lower_side-2-1", Kokkos::RangePolicy(Jx+4,c_Nx+4), KOKKOS_LAMBDA(const size_t i)
    {
        U(i,Jy+3,0) = (4.0*U(i,Jy+4,0)-U(i,Jy+5,0))/3.0;
        U(i,Jy+3,1) = (4.0*U(i,Jy+4,1)-U(i,Jy+5,1))/3.0;
        U(i,Jy+3,2) = 0;
        U(i,Jy+3,3) = (4.0*U(i,Jy+4,3)-U(i,Jy+5,3))/3.0;
    });
Kokkos::fence();
    //Kokkos::parallel_for( "wall_lower_side-2-2", Kokkos::RangePolicy(Jx+7,c_Nx+4), KOKKOS_LAMBDA(const size_t i)
    Kokkos::parallel_for( "wall_lower_side-2-2",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({Jx+7, Jy}, {c_Nx+4, Jy+3}),
    KOKKOS_LAMBDA(const size_t i, const size_t j)
    {
            U(i,j,0) =  U(i,2*(Jy+3)-j,0);
            U(i,j,1) =  U(i,2*(Jy+3)-j,1);
            U(i,j,2) = 0;
            U(i,j,3) =  U(i,2*(Jy+3)-j,3);
    });
Kokkos::fence();
    Kokkos::parallel_for( "wall_right_side-1", Kokkos::RangePolicy(4,Jy+4), KOKKOS_LAMBDA(const size_t j)
    {
        U(Jx+3,j,0) = U(Jx+2,j,0);
        U(Jx+3,j,1) = 0;
        U(Jx+3,j,2) = U(Jx+2,j,2);
        U(Jx+3,j,3) = U(Jx+2,j,3);
    });
Kokkos::fence();
    
    //Kokkos::parallel_for( "wall_right_side-2", Kokkos::RangePolicy(4,Jy+1), KOKKOS_LAMBDA(const size_t j)
    Kokkos::parallel_for( "wall_right_side-2",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({Jx+4, 4}, {Jx+7, Jy+1}),
    KOKKOS_LAMBDA(const size_t i, const size_t j)
    {
            U(i,j,0) =  U(2*(Jx+3)-i,j,0);
            U(i,j,1) = 0;
            U(i,j,2) =  U(2*(Jx+3)-i,j,2);
            U(i,j,3) =  U(2*(Jx+3)-i,j,3);
    });  
Kokkos::fence();
// - corner
    U(Jx+4,Jy+2,0) = (U(Jx+4,Jy+4,0)+U(Jx+2,Jy+2,0))/2.0;
    U(Jx+4,Jy+2,1) = (U(Jx+4,Jy+4,1)+U(Jx+2,Jy+2,1))/2.0;
    U(Jx+4,Jy+2,2) = (U(Jx+4,Jy+4,2)+U(Jx+2,Jy+2,2))/2.0;
    U(Jx+4,Jy+2,3) = (U(Jx+4,Jy+4,3)+U(Jx+2,Jy+2,3))/2.0;

    U(Jx+4,Jy+1,0) = U(Jx+2,Jy+1,0);
    U(Jx+4,Jy+1,1) =-U(Jx+2,Jy+1,1);
    U(Jx+4,Jy+1,2) = U(Jx+2,Jy+1,2);
    U(Jx+4,Jy+1,3) = U(Jx+2,Jy+1,3);

    U(Jx+4,Jy,0) = U(Jx+2,Jy,0);
    U(Jx+4,Jy,1) =-U(Jx+2,Jy,1);
    U(Jx+4,Jy,2) = U(Jx+2,Jy,2);
    U(Jx+4,Jy,3) = U(Jx+2,Jy,3);

    U(Jx+5,Jy,0) = U(Jx+1,Jy,0);
    U(Jx+5,Jy,1) =-U(Jx+1,Jy,1);
    U(Jx+5,Jy,2) = U(Jx+1,Jy,2);
    U(Jx+5,Jy,3) = U(Jx+1,Jy,3);

    U(Jx+5,Jy+2,0) = U(Jx+5,Jy+4,0);
    U(Jx+5,Jy+2,1) = U(Jx+5,Jy+4,1);
    U(Jx+5,Jy+2,2) =-U(Jx+5,Jy+4,2);
    U(Jx+5,Jy+2,3) = U(Jx+5,Jy+4,3);

    U(Jx+6,Jy+2,0) = U(Jx+6,Jy+4,0);
    U(Jx+6,Jy+2,1) = U(Jx+6,Jy+4,1);
    U(Jx+6,Jy+2,2) =-U(Jx+6,Jy+4,2);
    U(Jx+6,Jy+2,3) = U(Jx+6,Jy+4,3);

    U(Jx+6,Jy+1,0) = U(Jx+6,Jy+5,0);
    U(Jx+6,Jy+1,1) = U(Jx+6,Jy+5,1);
    U(Jx+6,Jy+1,2) =-U(Jx+6,Jy+5,2);
    U(Jx+6,Jy+1,3) = U(Jx+6,Jy+5,3);

    U(Jx+5,Jy+1,0) = (U(Jx+5,Jy+5,0)+U(Jx+1,Jy+1,0))/2.0;
    U(Jx+5,Jy+1,1) = (U(Jx+5,Jy+5,1)+U(Jx+1,Jy+1,1))/2.0;
    U(Jx+5,Jy+1,2) = (U(Jx+5,Jy+5,2)+U(Jx+1,Jy+1,2))/2.0;
    U(Jx+5,Jy+1,3) = (U(Jx+5,Jy+5,3)+U(Jx+1,Jy+1,3))/2.0;

    U(Jx+6,Jy,0) = U(Jx+3,Jy+3,0);
    U(Jx+6,Jy,1) =-U(Jx+3,Jy+3,1);
    U(Jx+6,Jy,2) =-U(Jx+3,Jy+3,2);
    U(Jx+6,Jy,3) = U(Jx+3,Jy+3,3);

    
    //Kokkos::parallel_for( "ghost-1", Kokkos::RangePolicy(Jx+4,Jx+7), KOKKOS_LAMBDA(const size_t i)
    Kokkos::parallel_for( "ghost-1",Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({Jx+4, 0}, {Jx+7, 3}),
    KOKKOS_LAMBDA(const size_t i, const size_t j)
    {
            U(i,j,0) = U(2*(Jx+3)-i,6-j,0);
            U(i,j,1) = U(2*(Jx+3)-i,6-j,1);
            U(i,j,2) = U(2*(Jx+3)-i,6-j,2);
            U(i,j,3) = U(2*(Jx+3)-i,6-j,3);
    });
    Kokkos::fence();

    Kokkos::parallel_for( "ghost-2", Kokkos::RangePolicy(0,3), KOKKOS_LAMBDA(const size_t j)
    {
        U(Jx+3,j,0) = U(Jx+3,6-j,0);
        U(Jx+3,j,1) = U(Jx+3,6-j,1);
        U(Jx+3,j,2) =-U(Jx+3,6-j,2);
        U(Jx+3,j,3) = U(Jx+3,6-j,3);
    });
    Kokkos::fence();

    Kokkos::parallel_for( "ghost-2", Kokkos::RangePolicy(Jx+4,Jx+7), KOKKOS_LAMBDA(const size_t i)
    {
        U(i,3,0) = U(2*(Jx+3)-i,3,0);
        U(i,3,1) =-U(2*(Jx+3)-i,3,1);
        U(i,3,2) = U(2*(Jx+3)-i,3,2);
        U(i,3,3) = U(2*(Jx+3)-i,3,3);
    });


    Kokkos::fence();

}
#endif
