! Controlpara_mod.f90
!
!---------------------------------------------------------------------------------------------------------------
module Controlpara_mod   !control parameters

  use Constant_mod
  implicit none

! - grid parameters
integer,parameter         ::          Nx = 1010,Ny = 242!Nx = 505,Ny = 121  !cell number
integer,parameter         ::         nghost = 3          !number of ghost cells
integer,parameter         ::         is = 0 - nghost
integer,parameter         ::         ie = Nx + nghost
integer,parameter         ::         js = 0 - nghost
integer,parameter         ::         je = Ny + nghost
integer,parameter         ::         xx = ie - is
integer,parameter         ::         yy = je - js



real(kreal),parameter     ::         Lx = 4.0, Ly = 1.0

! - grid parameters
real(kreal),parameter     ::         R = 287.0            !the ideal gas constant,J/(kg*K)
real(kreal),parameter     ::         Gamma = 1.4          !specific heat ratio

! - inflow information
real(kreal),parameter     ::         Mach_inf = 3.0       !the mach number
real(kreal),parameter     ::         Rho_inf = 1.0        !density,kg/m^3
real(kreal),parameter     ::         P_inf = 0.71429      !pressure,N/m^2

! - iteration parameters
integer,parameter         ::         itermax = 100   !number of iteration20000
real(kreal),parameter     ::         CFL = 0.1            !CFL number

! - equation parameters
integer,parameter         ::         Nvar = 4             !number of equations

    end module
!------------------------------------------------------------------------------------------------------------------