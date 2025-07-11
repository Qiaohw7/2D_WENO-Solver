// control parameters
#ifndef Controlpara_HPP
#define Controlpara_HPP

// - grid parameters
constexpr int c_Nx = 1010, c_Ny = 242  ;  //c_Nx = 505, c_Ny = 121
constexpr int c_nghost = 3;          // number of ghost cells
constexpr int c_is = 0 - c_nghost;
constexpr int c_ie = c_Nx + c_nghost;
constexpr int c_js = 0 - c_nghost;
constexpr int c_je = c_Ny + c_nghost;
constexpr int c_xx = c_ie - c_is;
constexpr int c_yy = c_je - c_js;

constexpr double c_Lx = 4.0, c_Ly = 1.0;

// - grid parameters
constexpr double c_R = 287.0;            // the ideal gas constant,J/(kg*K)
constexpr double c_Gamma = 1.4;          //specific heat ratio

// - inflow information
constexpr double c_Mach_inf = 3.0;       //the mach number
constexpr double c_Rho_inf = 1.0;        //density,kg/m^3
constexpr double c_P_inf = 0.71429;      //pressure,N/m^2

// - iteration parameters
constexpr int c_itermax = 1000;   // number of iteration20000
constexpr double c_CFL = 0.1 ;   //CFL number

// - equation parameters
constexpr int c_Nvar = 4 ;            //number of equations

#endif