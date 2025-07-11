//C++ interface

#include <Kokkos_Core.hpp>
#include <cstdio>
#include "flcl-cxx.hpp"
#include "Controlpara.hpp"
#include "BoundUpdate_mod.hpp"
#include "Timestep.hpp"
#include "InviscidFlux.hpp"

static Kokkos::Timer timer;   //make sure kokkos::Timer can be used
using view_type = flcl::view_r64_3d_t;
using view_type_r1d = flcl ::view_r64_1d_t;

extern "C" {

// 1 - Initialize
    void c_Initialize( view_type **v_U)
    {
        using flcl::view_from_ndarray;

        view_type U = **v_U;
    
    // - local variables
        
    // - get data from controlpara.hpp
        double T_inf = c_P_inf / (c_Rho_inf*c_R);
        double U_inf = c_Mach_inf * sqrt(c_Gamma*c_R*T_inf);
        double V_inf = 0.0;

    // - init field
        // #if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
       Kokkos::parallel_for("Initialize",  Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({0, 0}, {c_Nx+7, c_Ny+7}), 
            KOKKOS_LAMBDA (const int i, const int j) 
        {
                U(i,j,0) = c_Rho_inf;
                U(i,j,1) = c_Rho_inf*U_inf;
                U(i,j,2) = c_Rho_inf*V_inf;
                U(i,j,3) = c_P_inf/(c_Gamma-1) + 0.5*c_Rho_inf*(U_inf*U_inf+V_inf*V_inf); 
            
        });
        // #endif  
        Kokkos::fence();

        // Kokkos::parallel_for("ini2",  Kokkos::MDRangePolicy<Kokkos::Rank<3>>({c_Nx/4+3, 3, 0}, {c_Nx+4, c_Ny/5+4, c_Nvar}), 
        //     KOKKOS_LAMBDA (const size_t i, const size_t j, const size_t n) 
        // {
        //         U(i,j,n) = 0.001;
        // });

        // Kokkos::fence();

    // - init bound
        BoundUpdate( U );
    }

// --------------------------------------------------------------------------------------------------
// 2 - Runge_Kutta
    void c_Runge_Kutta( view_type &U,InviscidFluxCalculator &INVIS, int &iter, double &Time)
    {
        using flcl::view_from_ndarray;        
        
        // local value
        int RkStage = 3;
        double dt;
                   
        double RKalfa[3];       // view_type_r1d RKalfa("RKalfa", 3);
        double RKbeta[3];       // view_type_r1d RKbeta("RKbeta", 3);
        view_type U0("U0", c_Nx+7, c_Ny+7, c_Nvar );
        view_type RHS("RHS", c_Nx+7, c_Ny+7, c_Nvar );
        Kokkos::deep_copy(U0, 0.0);
        Kokkos::deep_copy(RHS, 0.0);
        RKalfa[0] = 1.0 ;      RKbeta[0] = 1.0;
        RKalfa[1] = 3.0/4.0 ;  RKbeta[1] = 1.0/4.0;
        RKalfa[2] = 1.0/3.0 ;  RKbeta[2] = 2.0/3.0;

        // global step
        Timestep( U, dt );

        Time = Time + dt;   
        std::cout << "iter = " << iter << "   Time = " << Time << "   dt = " << dt << std::endl;

        Kokkos::parallel_for("copyU0",  Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<3>>({0, 0, 0}, {c_Nx+7, c_Ny+7, c_Nvar}), 
            KOKKOS_LAMBDA (const int i, const int j, const int n) 
        {
                U0(i,j,n) = U(i,j,n);
        });

        Kokkos::fence();

//========================================================
// auto RHS_mirror = Kokkos::create_mirror_view(RHS);
// Kokkos::deep_copy(RHS_mirror, U0);
// auto dim1 = RHS.extent(0); // 第一维度的大小
    
//     // 遍历Fr_mirror来输出数据
// for(size_t i = 2; i < (dim1/50); ++i) {    
//             std::cout << "RHS(" << i <<  ") = " << RHS_mirror(i,4,2) << std::endl;     
//     }    
//--------------------------------------------------------------------------
        for(int Kstep = 0; Kstep < RkStage; Kstep++)
        {
            INVIS.InviscidFlux_x( U,RHS);
            // Kokkos::fence();
            INVIS.InviscidFlux_y( U,RHS);
        // -----------------------------------------------------------------
            // auto U_mirror = Kokkos::create_mirror_view(U);
            // Kokkos::deep_copy(U_mirror, U);
            // auto dim1 = U.extent(0); // 第一维度的大小
                
            //     // 遍历Fr_mirror来输出数据
            // for(size_t i = 1; i < (dim1/20); ++i) {    
            //             std::cout << "U(" << i <<  ") = " << U_mirror(i,4,1) << std::endl;     
            //     }   
        // -----------------------------------------------------------------  
            Kokkos::parallel_for("Runge", Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<3>>({0, 0, 0}, {c_Nx+7, c_Ny+7, c_Nvar}), 
            KOKKOS_LAMBDA (const int i, const int j, const int n) 
                {
                    if(Kstep == 0)
                    {     
                        U(i,j,n)=RKalfa[Kstep]* U0(i,j,n) + RKbeta[Kstep] * dt * RHS(i,j,n) ;
                    }
                    else
                    {
                        U(i,j,n)=RKalfa[Kstep]*U0(i,j,n) + RKbeta[Kstep]*(U(i,j,n)+dt*RHS(i,j,n));
                    }
                }); 

            BoundUpdate( U ); 

        }

        Kokkos::fence();

        U0 = view_type();
        
    }
// ------------------------------------------------------------------------------------------------
// 3 - Iteration
    void c_Iteration(view_type **v_U, double *v_Time  )
    {
        using flcl::view_from_ndarray;

         // inout variables
        view_type U = **v_U;
        // view_type RHS = **v_RHS;
        // int iter = *v_iter;              
        double Time = *v_Time;      //要修改的引用需要加上地址索引
        // double tt = 0.0;
        
        InviscidFluxCalculator INVIS;  //减少多次创建数据类型
        INVIS.initialize();

        for(int iter = 1; iter<=c_itermax; iter++)
        {
            c_Runge_Kutta(U ,INVIS, iter, Time);
            // tt = iter + tt;
        }
        // std::cout << "c_TIme = " << Time << std::endl;
        *v_Time=Time;
    }
// --------------------------------------------------------------------------------------------------
// 00 - Record TIME (now is very simple, can add more features later)
    void c_reset_timer()
    {
        timer.reset();
    }

    double c_get_time_fortranKokkos() 
    {
        double c_time = timer.seconds();
        return c_time;
    }

}