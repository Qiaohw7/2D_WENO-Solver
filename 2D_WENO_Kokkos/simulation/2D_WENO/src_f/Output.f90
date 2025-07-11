! Output_mod.f90
!
!-------------------------------------------------------------------------------------------------------
subroutine Output( iter, Time, U)
    
    use Controlpara_mod
    implicit none 

! - inout variables
    integer,intent(in)        :: iter
    real(kreal),intent(in)    :: Time
    real(kreal),intent(in)    :: U(is:ie,js:je,Nvar)
! - local variables
    integer                ::    i, j, n,NST, Jy, Jx
    real(kreal)            ::    rhol,ul,vl,El,pl
    real(kreal)            ::    dx, dy
    character(len=15)    ::    Fname
    
    NST = 50
    n = iter/NST
    Jx = Nx/4
    Jy = Ny/5
    
    if( mod(iter,NST)/=0 )then
        return
    end if
    
    dx = Lx / Nx
    dy = Ly / Ny
    
    
    Fname='result  .dat'
    write(Fname(7:8),'(I2.2)') n
    
    open (1,File=Fname,status='unknown')
        write(1,*)'TITLE = "RESULT_',Time,'s"'
        write(1,*)'VARIABLES = "x" "y" "rho" "u" "v" "p" "E" '
        write(1,*)'ZONE T="Zone 1"'
        write(1,*)'I=',Nx+1,'J=',Ny+1,'K=',1,'ZONETYPE=Ordered'
        write(1,*)'DATAPACKING=POINT'
        
        do j=0,Ny
        do i=0,Nx
            if (i > Jx.and. j < Jy) then
            rhol = 0.0
            ul   = 0.0
            vl   = 0.0
            El     = 0.0
            pl     = 0.0
            else
            rhol = U(i,j,1)
            ul   = U(i,j,2)/rhol
            vl   = U(i,j,3)/rhol
            El     = U(i,j,4)
            pl     = (Gamma-1)*(U(i,j,4)-0.5*rhol*(ul*ul+vl*vl))
            end if
            write(1,200) i*dx,j*dy,rhol,ul,vl,pl,El
        end do
        end do
        
    close(1)
    !rint *,Jx,Jy
200    format(D20.10,D20.10,D20.10,D20.10,D20.10,D20.10,D20.10)

    
end subroutine
!-------------------------------------------------------------------------------------------------------
