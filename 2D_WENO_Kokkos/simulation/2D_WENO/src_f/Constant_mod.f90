! Constant_mod.f90 -  module for all constants or default values in Euler2D
!
!---------------------------------------------------------------------------------
module Constant_mod    

    implicit none

! - Kind of data type
    integer,parameter        ::       single = kind(1.E-8)
    integer,parameter        ::       double = kind(1.D-8)  
    integer,parameter        ::       kreal  = double
    
end module
!----------------------------------------------------------------------------------