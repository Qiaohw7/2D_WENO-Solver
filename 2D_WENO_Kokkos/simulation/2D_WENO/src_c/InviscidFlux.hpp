#ifndef InviscidFlux_HPP
#define InviscidFlux_HPP

#include <Kokkos_Core.hpp>
#include <cstdio>
#include "flcl-cxx.hpp"
#include "Controlpara.hpp"

class InviscidFluxCalculator{

private:

    using exec_space = Kokkos::DefaultExecutionSpace;
    using view_type = flcl::view_r64_3d_t;
    using view_type_r1d = flcl ::view_r64_1d_t;
    using view_type_r2d = flcl ::view_r64_2d_t;

    double Lx = c_Lx;
    double Ly = c_Ly;

    view_type LLL;
    
    view_type Hp1;
    view_type Hp2;
    view_type Hp3;
    view_type Hn1;
    view_type Hn2;
    view_type Hn3;
    view_type ISl1;
    view_type ISl2;
    view_type ISl3;
    view_type ISr1;
    view_type ISr2;
    view_type ISr3;
    view_type omegal1;
    view_type omegal2;
    view_type omegal3;
    view_type omegar1;
    view_type omegar2;
    view_type omegar3;
    view_type alphl1;
    view_type alphl2;
    view_type alphl3;
    view_type alphr1;
    view_type alphr2;
    view_type alphr3;

    view_type Fr_x;
    view_type Fl_x;
    view_type Fr_y;
    view_type Fl_y;
    view_type Gl;
    view_type Gr;

    view_type_r2d rhol;
    view_type_r2d ul;
    view_type_r2d vl;
    view_type_r2d pl;
    view_type_r2d hl;
    view_type_r2d al;
                    


public:

    InviscidFluxCalculator():LLL("LLL",c_Nx+7,c_Ny+7, 6),
                            Hp1("Hp1",c_Nx+7,c_Ny+7,c_Nvar),
                            Hp2("Hp2",c_Nx+7,c_Ny+7,c_Nvar),
                            Hp3("Hp3",c_Nx+7,c_Ny+7,c_Nvar),
                            Hn1("Hn1",c_Nx+7,c_Ny+7,c_Nvar),
                            Hn2("Hn2",c_Nx+7,c_Ny+7,c_Nvar),
                            Hn3("Hn3",c_Nx+7,c_Ny+7,c_Nvar),
                            ISl1("ISl1",c_Nx+7,c_Ny+7,c_Nvar),
                            ISl2("ISl2",c_Nx+7,c_Ny+7,c_Nvar),
                            ISl3("ISl3",c_Nx+7,c_Ny+7,c_Nvar),
                            ISr1("ISr1",c_Nx+7,c_Ny+7,c_Nvar),
                            ISr2("ISr2",c_Nx+7,c_Ny+7,c_Nvar),
                            ISr3("ISr3",c_Nx+7,c_Ny+7,c_Nvar),
                            omegal1("omegal1",c_Nx+7,c_Ny+7,c_Nvar),
                            omegal2("omegal2",c_Nx+7,c_Ny+7,c_Nvar),
                            omegal3("omegal3",c_Nx+7,c_Ny+7,c_Nvar),
                            omegar1("omegar1",c_Nx+7,c_Ny+7,c_Nvar),
                            omegar2("omegar2",c_Nx+7,c_Ny+7,c_Nvar),
                            omegar3("omegar3",c_Nx+7,c_Ny+7,c_Nvar),
                            alphl1("arphl1",c_Nx+7,c_Ny+7,c_Nvar),
                            alphl2("arphl2",c_Nx+7,c_Ny+7,c_Nvar),
                            alphl3("arphl3",c_Nx+7,c_Ny+7,c_Nvar),
                            alphr1("arphr1",c_Nx+7,c_Ny+7,c_Nvar),
                            alphr2("arphr2",c_Nx+7,c_Ny+7,c_Nvar),
                            alphr3("arphr3",c_Nx+7,c_Ny+7,c_Nvar),
                           
                            Fl_x("Fl",c_Nx+7,c_Ny+7,c_Nvar),
                            Fr_x("Fr",c_Nx+7,c_Ny+7,c_Nvar),
                            Fl_y("Fl",c_Nx+7,c_Ny+7,c_Nvar),
                            Fr_y("Fr",c_Nx+7,c_Ny+7,c_Nvar),
                            Gl("Gl",c_Nx+7,c_Ny+7,c_Nvar),
                            Gr("Gr",c_Nx+7,c_Ny+7,c_Nvar),
                            
                            rhol("rhol",c_Nx+7,c_Ny+7),
                            ul("ul",c_Nx+7,c_Ny+7),
                            vl("vl",c_Nx+7,c_Ny+7),
                            pl("pl",c_Nx+7,c_Ny+7),
                            hl("hl",c_Nx+7,c_Ny+7),
                            al("al",c_Nx+7,c_Ny+7){

                            }

void initialize(){

    Kokkos::deep_copy(LLL, 0.0);
    Kokkos::deep_copy(Fl_x, 0.0);  Kokkos::deep_copy(Fr_x, 0.0);
    Kokkos::deep_copy(Fl_y, 0.0);  Kokkos::deep_copy(Fr_y, 0.0);
    Kokkos::deep_copy(Gl, 0.0);  Kokkos::deep_copy(Gr, 0.0);
    Kokkos::deep_copy(Hp1, 0.0);  Kokkos::deep_copy(Hn1, 0.0);
    Kokkos::deep_copy(Hp2, 0.0);  Kokkos::deep_copy(Hn2, 0.0);
    Kokkos::deep_copy(Hp3, 0.0);  Kokkos::deep_copy(Hn3, 0.0);
    Kokkos::deep_copy(ISl1, 0.0);  Kokkos::deep_copy(ISr1, 0.0);
    Kokkos::deep_copy(ISl2, 0.0);  Kokkos::deep_copy(ISr2, 0.0);
    Kokkos::deep_copy(ISl3, 0.0);  Kokkos::deep_copy(ISr3, 0.0);
    Kokkos::deep_copy(omegal1, 0.0);  Kokkos::deep_copy(omegar1, 0.0);
    Kokkos::deep_copy(omegal2, 0.0);  Kokkos::deep_copy(omegar2, 0.0);
    Kokkos::deep_copy(omegal3, 0.0);  Kokkos::deep_copy(omegar3, 0.0);
    Kokkos::deep_copy(alphl1, 0.0);  Kokkos::deep_copy(alphr1, 0.0);
    Kokkos::deep_copy(alphl2, 0.0);  Kokkos::deep_copy(alphr2, 0.0);
    Kokkos::deep_copy(alphl3, 0.0);  Kokkos::deep_copy(alphr3, 0.0);
    Kokkos::deep_copy(rhol, 0.0);   Kokkos::deep_copy(ul, 0.0);
    Kokkos::deep_copy(vl, 0.0);    Kokkos::deep_copy(pl, 0.0);
    Kokkos::deep_copy(hl, 0.0);     Kokkos::deep_copy(al, 0.0);

    // auto Fr_mirror = Kokkos::create_mirror_view(Fr);
    // Kokkos::deep_copy(Fr_mirror, Flux);
    // auto dim0 = Flux.extent(0); // 第一维度的大小
    
    // // 遍历Fr_mirror来输出数据
    // for(size_t i = 0; i < dim0; ++i) {
        
    //         std::cout << "Fr(" << i <<  ") = " << Fr_mirror(i) << std::endl;
        
    // }

}

// --------------------------------------------------------------------------------------------------
// KOKKOS_INLINE_FUNCTION
// // KOKKOS_FUNCTION
// void Eigenvalue(double ul, double al, view_type_r1d LLL, int sign){
    
//     double Epsilon = 1e-4;

//     LLL(0) = ul;
//     LLL(1) = ul;
//     LLL(2) = ul - al;
//     LLL(3) = ul + al;

//     for(int n = 0; n < c_Nvar; n++)
//     {
//         LLL(n) = 0.5*(LLL(n) + sign*sqrt(Kokkos::pow(LLL(n),2) + Kokkos::pow(Epsilon,2)));
//     }
// }

// KOKKOS_INLINE_FUNCTION
// void Fluxsplit( double rhol, double ul, double vl, double hl, double al, int Direction, int sign, view_type_r1d Flux, view_type_r1d LLL ){
//     if(Direction == 1)
//     {
//         Eigenvalue(ul, al, LLL, sign);

//         Flux(0) = rhol/(2*c_Gamma)*(2*(c_Gamma-1)*LLL(0)+LLL(2)+LLL(3));
//         Flux(1) = rhol/(2*c_Gamma)*(2*(c_Gamma-1)*ul*LLL(0)+(ul-al)*LLL(2)+(ul+al)*LLL(3));
//         Flux(2) = rhol/(2*c_Gamma)*(2*(c_Gamma-1)*vl*LLL(1)+vl*LLL(2)+vl*LLL(3));
//         Flux(3) = rhol/(2*c_Gamma)*((c_Gamma-1)*(ul*ul-vl*vl)*LLL(0)+2*(c_Gamma-1)*vl*vl*LLL(1)+(hl-al*ul)*LLL(2)+(hl+al*ul)*LLL(3));
//     }
//     else if(Direction == 2)
//     {
//         Eigenvalue(ul, al, LLL, sign);

//         Flux(0) = rhol/(2*c_Gamma)*(2*(c_Gamma-1)*LLL(1)+LLL(2)+LLL(3));
//         Flux(1) = rhol/(2*c_Gamma)*(2*(c_Gamma-1)*ul*LLL(0)+ul*LLL(2)+ul*LLL(3));
//         Flux(2) = rhol/(2*c_Gamma)*(2*(c_Gamma-1)*vl*LLL(1)+(vl-al)*LLL(2)+(vl+al)*LLL(3));
//         Flux(3) = rhol/(2*c_Gamma)*((c_Gamma-1)*(vl*vl-ul*ul)*LLL(1)+2*(c_Gamma-1)*ul*ul*LLL(0)+(hl-al*vl)*LLL(2)+(hl+al*vl)*LLL(3));
//     }

// }

void InviscidFlux_x( view_type &U, view_type &RHS);


void InviscidFlux_y( view_type &U, view_type &RHS);

};

#endif