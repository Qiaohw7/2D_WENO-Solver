module WENO5_f_mod
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env

    use :: flcl_mod 
    
    implicit none

    public

    ! interface
    !     subroutine f_BoundUpdate(nd_array_U)&
    !       & bind(c, name='c_BoundUpdate')
    !         use, intrinsic :: iso_c_binding
    !         use :: flcl_ndarray_mod
    !         implicit none
    !         type(nd_array_t) :: nd_array_U
    !     end subroutine f_BoundUpdate
    ! end interface

    interface
        subroutine f_Initialize( U ) &
          & bind(c, name='c_Initialize')
            use, intrinsic :: iso_c_binding
            use :: flcl_mod
            
            type(c_ptr),intent(in)  :: U
        end subroutine f_Initialize
    end interface

    interface
        subroutine f_Iteration( U,  Time )&
          & bind(c, name='c_Iteration')
            use, intrinsic :: iso_c_binding
            use :: flcl_mod
            
            ! integer, intent(in)         :: iter
            real(c_double), intent(in)  :: Time
            type(c_ptr),intent(in)      :: U
            ! type(c_ptr),intent(in)      :: RHS
        end subroutine f_Iteration
    end interface

    ! - time-kokkos
    interface
        subroutine f_reset_timer() bind(C, name="c_reset_timer")
          use, intrinsic :: iso_c_binding
          
        end subroutine f_reset_timer
      end interface

      interface
        function f_get_time_fortranKokkos() result(time) bind(C, name="c_get_time_fortranKokkos")
          use, intrinsic :: iso_c_binding
          implicit none
          real(c_double) :: time
        end function f_get_time_fortranKokkos
      end interface

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! subroutine BoundUpdate(U)
    !     use, intrinsic :: iso_c_binding
    !     use :: flcl_ndarray_mod
    !     implicit none
    !     real(c_double), dimension(:,:,:), intent(inout) :: U

    !     call f_BoundUpdate(to_nd_array(U))
    ! end subroutine BoundUpdate

    subroutine Initialize(U)
        use, intrinsic :: iso_c_binding
        use :: flcl_mod
        implicit none

        type(view_r64_3d_t), intent(inout)  :: U
        call f_Initialize( U%ptr() )             !pass the ptr adress to C++ memory
    end subroutine Initialize

    subroutine Iteration( U, Time  )
        use, intrinsic :: iso_c_binding
        use :: flcl_mod
        implicit none

        ! integer, intent(inout)              :: iter
        real(c_double), intent(inout)       :: Time
        type(view_r64_3d_t), intent(inout)  :: U
        ! type(view_r64_3d_t), intent(inout)     :: RHS
        call f_Iteration( U%ptr(), Time)
    end subroutine Iteration

    subroutine reset_timer()
        use, intrinsic :: iso_c_binding
        implicit none

        call f_reset_timer()
    end subroutine reset_timer

     function get_time_fortranKokkos() result(time)
        use, intrinsic :: iso_c_binding
        implicit none

        real(c_double) :: c_time
        real           :: time
        c_time = f_get_time_fortranKokkos()
        time = real(c_time)
    end function get_time_fortranKokkos




end module WENO5_f_mod