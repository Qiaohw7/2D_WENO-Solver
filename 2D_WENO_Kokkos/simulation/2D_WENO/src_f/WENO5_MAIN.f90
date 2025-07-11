! WENO5_2DSolver.f90
!
!----------------------------------------------------------------------------------------------------------------------
program WENO5_MAIN

    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env

    use   :: Controlpara_mod
    use   :: flcl_util_kokkos_mod
    ! use   :: flcl_view_mod
    use   :: flcl_mod
    use   :: WENO5_f_mod

    implicit none

    integer                     :: iter
    integer(8)                  :: e0, e1, e2
    real(c_double)              :: Time
    real(REAL64), pointer, dimension(:,:,:)       ::  U      !test the difference between REAL64 and c_double
    ! real(REAL64), pointer, dimension(:, :, :)     ::  RHS
    real(c_double)              ::  compute_time

    type(view_r64_3d_t) :: v_U
    ! type(view_r64_3d_t) :: v_RHS

    e0 = xx+1  !fortan is:ie 声明的数组比这样大1,要加上+1
    e1 = yy+1
    e2 = Nvar

!- initialize kokkos
  call kokkos_initialize()
  write(*,*)'initializing kokkos successfully'

!- allocate views
  call kokkos_allocate_v_r64_3d( U, v_U, 'U', e0,e1,e2 )     !具体的维度和精度可查flcl-view-f.f90
  ! call kokkos_allocate_v_r64_3d( RHS, v_RHS, 'RHS', e0,e1,e2)
  write(*,*)'allocating kokkos views successfully'
  ! call kokkos_allocate_view( U, v_U, 'U', xx,yy,Nvar )
  ! call kokkos_allocate_view( RHS, v_RHS, 'RHS', xx,yy,Nvar)      
!---------------------------------------------------------------------------------------------
! - compute start
  write(*,*)' compute real time' 
  call reset_timer()
! - initialize 2D flow
  U = 0.0
  ! RHS = 0.0
  Time  = 0.0
  call Initialize( v_U )
  write(*,*)'-------------------------------------------------------------------------------------'
  write(*,*)'flow Initialize successfully'

! ================================================
! - compute start
  write(*,*)' compute real time' 
  call reset_timer() 
      
  ! call Iteration( v_U, v_RHS, Time)   !计算为C++部分,传递 v_ 参数
  call Iteration( v_U, Time)

    ! do iter=1,itermax  
      
  call Output( iter, Time, U )      !输出为Fortran部分,传递原有参数

    ! end do

  compute_time = get_time_fortranKokkos()
  write(*,*)'compute time = ',compute_time

!---------------------------------------------------------------------------------------------
! - deallocate memory  
  write(*,*)'deallocate kokkos views'
  call kokkos_deallocate_v_r64_3d( U, v_U )
  ! call kokkos_deallocate_v_r64_3d( RHS, v_RHS )
  ! call kokkos_deallocate_view( U, v_U )
  ! call kokkos_deallocate_view( RHS, v_RHS )
  
!- finalize kokkos
  write(*,*)'--------------------------------------------------------------------------'
  write(*,*)'finalizing kokkos'
  call kokkos_finalize()

end program WENO5_MAIN
  