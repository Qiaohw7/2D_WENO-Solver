#include <Kokkos_Core.hpp>
#include <cstdio>
#include <Kokkos_MathematicalFunctions.hpp>
#include "flcl-cxx.hpp"
#include "Controlpara.hpp"
#include "InviscidFlux.hpp"

void InviscidFluxCalculator::InviscidFlux_x( view_type &U, view_type &RHS){
    using flcl::view_from_ndarray;

    double dx = Lx/c_Nx;
    
    // int Direction = 1;
    double Epsilon = 1e-6;
    double IS1, IS2;
    double Cl[3], Cr[3];
    double Rbeta[9], Lbeta[9];
    
    // Initialize value
    IS1 = 13.0/12.0;        IS2 = 1.0/4.0;
    Cl[0] = 3.0/10.0;       Cr[0] = 1.0/10.0;
    Cl[1] = 3.0/5.0;        Cr[1] = 3.0/5.0;
    Cl[2] = 1.0/10.0;       Cr[2] = 3.0/10.0;
    Rbeta[0] =  1.0/3.0;   Lbeta[0] = -1.0/6.0;
    Rbeta[1] = -7.0/6.0;   Lbeta[1] =  5.0/6.0;
    Rbeta[2] =  11.0/6.0;  Lbeta[2] =  1.0/3.0;
    Rbeta[3] = -1.0/6.0;   Lbeta[3] =  1.0/3.0;
    Rbeta[4] =  5.0/6.0;   Lbeta[4] =  5.0/6.0;
    Rbeta[5] =  1.0/3.0;   Lbeta[5] = -1.0/6.0;
    Rbeta[6] =  1.0/3.0;   Lbeta[6] =  11.0/6.0;
    Rbeta[7] =  5.0/6.0;   Lbeta[7] = -7.0/6.0;
    Rbeta[8] = -1.0/6.0;   Lbeta[8] =  1.0/3.0;
    
    auto lLLL = LLL;
    auto lFr_x = Fr_x; auto lFl_x = Fl_x;

    auto lGl = Gl; auto lGr = Gr;
    auto lHp1= Hp1; auto lHn1 = Hn1;
    auto lHp2 = Hp2; auto lHn2 = Hn2;
    auto lHp3 = Hp3; auto lHn3 = Hn3;
    auto lISl1 = ISl1; auto lISr1 = ISr1;
    auto lISl2 = ISl2; auto lISr2 = ISr2;
    auto lISl3 = ISl3; auto lISr3 = ISr3;
    auto lomegal1 = omegal1; auto lomegar1 = omegar1;
    auto lomegal2 = omegal2; auto lomegar2 = omegar2;
    auto lomegal3 = omegal3; auto lomegar3 = omegar3;
    auto lalphl1 = alphl1;  auto lalphr1 = alphr1;
    auto lalphl2 = alphl2;  auto lalphr2 = alphr2;
    auto lalphl3 = alphl3;  auto lalphr3 = alphr3;

    auto lrhol = rhol;  auto lal = al;
    auto lul = ul;  auto lvl = vl;
    auto lhl = hl;  auto lpl = pl;
    
// - SW split
    Kokkos::parallel_for("SW split", Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({0, 0}, {c_Nx+7, c_Ny+7}), 
               KOKKOS_LAMBDA (const int i, const int j) 
            {
                    lrhol(i,j) = U(i,j,0);
                    lul(i,j)= U(i,j,1)/lrhol(i,j);
                    lvl(i,j)   = U(i,j,2)/lrhol(i,j);
                    lpl(i,j)   = (c_Gamma-1)*(U(i,j,3)-0.5*lrhol(i,j)*(lul(i,j)*lul(i,j)+lvl(i,j)*lvl(i,j)));
                    lhl(i,j) = (U(i,j,3)+lpl(i,j))/lrhol(i,j);
                    if(lpl(i,j) < 0.0)
                    {
                        lpl(i,j)=1.e-4;
                    }
                    lal(i,j) = sqrt(fabs(c_Gamma*lpl(i,j)/lrhol(i,j)));
                    
                    // lLLL(i,j,0) = 0.5*(lul(i,j) + sqrt(fabs(lul(i,j)*lul(i,j)+0.0001*0.0001)) );
                    // lLLL(i,j,1) = 0.5*(lul(i,j) - sqrt(fabs(lul(i,j)*lul(i,j)+0.0001*0.0001)) );
                    // lLLL(i,j,2) = 0.5*(lul(i,j)-lal(i,j)) + sqrt(fabs((lul(i,j)-lal(i,j))*(lul(i,j)-lal(i,j))+0.0001*0.0001));
                    // lLLL(i,j,3) = 0.5*(lul(i,j)-lal(i,j)) - sqrt(fabs((lul(i,j)-lal(i,j))*(lul(i,j)-lal(i,j))+0.0001*0.0001));
                    // lLLL(i,j,4) = 0.5*(lul(i,j)+lal(i,j)) + sqrt(fabs((lul(i,j)+lal(i,j))*(lul(i,j)+lal(i,j))+0.0001*0.0001));
                    // lLLL(i,j,5) = 0.5*(lul(i,j)+lal(i,j)) - sqrt(fabs((lul(i,j)+lal(i,j))*(lul(i,j)+lal(i,j))+0.0001*0.0001));

                    // lFr_x(i, j, 0) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lLLL(i,j,0)+lLLL(i,j,2)+lLLL(i,j,4));
                    // lFr_x(i, j, 1) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lul(i,j)*lLLL(i, j, 0)+(lul(i,j)-lal(i,j))*lLLL(i, j, 2)+(lul(i,j)+lal(i,j))*lLLL(i, j, 4));
                    // lFr_x(i, j, 2) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lvl(i,j)*lLLL(i, j, 0)+lvl(i,j)*lLLL(i, j, 2)+lvl(i,j)*lLLL(i, j, 4));
                    // lFr_x(i, j, 3) = lrhol(i,j)/(2*c_Gamma)* ((c_Gamma-1)*(lvl(i,j)*lvl(i,j)+lul(i,j)*lul(i,j))*lLLL(i, j, 0)+(lhl(i,j)-lul(i,j)*lal(i,j))*lLLL(i,j,2)+(lhl(i,j)+lvl(i,j)*lal(i,j))*lLLL(i,j,4));

                    // lFl_x(i, j, 0) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lLLL(i, j, 1)+lLLL(i, j, 3)+lLLL(i, j, 5));
                    // lFl_x(i, j, 1) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lul(i,j)*lLLL(i, j, 1)+(lul(i,j)-lal(i,j))*lLLL(i, j, 3)+(lul(i,j)+lal(i,j))*lLLL(i, j, 5));
                    // lFl_x(i, j, 2) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lvl(i,j)*lLLL(i, j, 1)+lvl(i,j)*lLLL(i, j, 3)+lvl(i,j)*lLLL(i, j, 5));
                    // lFl_x(i, j, 3) = lrhol(i,j)/(2*c_Gamma)* ((c_Gamma-1)*(lvl(i,j)*lvl(i,j)+lul(i,j)*lul(i,j))*lLLL(i, j, 1)+(lhl(i,j)-lul(i,j)*lal(i,j))*lLLL(i,j,3)+(lhl(i,j)+lul(i,j)*lal(i,j))*lLLL(i,j,5));
                      
                    lLLL(i, j, 0) = lul(i,j);
                    lLLL(i, j, 1) = lul(i,j);
                    lLLL(i, j, 2) = lul(i,j) - lal(i,j);
                    lLLL(i, j, 3) = lul(i,j) + lal(i,j);
                                                                                               
                    for(int n = 0; n < c_Nvar; n++)
                    {                                                                          
                        lLLL(i, j, n) = 0.5*(lLLL(i, j, n) + sqrt( fabs(Kokkos::pow(lLLL(i, j, n),2) + Kokkos::pow(0.0001,2)) ) );
                    }

                    lFr_x(i, j, 0) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lLLL(i, j, 0)+lLLL(i, j, 2)+lLLL(i, j, 3));
                    lFr_x(i, j, 1) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lul(i,j)*lLLL(i, j, 0)+(lul(i,j)-lal(i,j))*lLLL(i, j, 2)+(lul(i,j)+lal(i,j))*lLLL(i, j, 3));
                    lFr_x(i, j, 2) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lvl(i,j)*lLLL(i, j, 1)+lvl(i,j)*lLLL(i, j, 2)+lvl(i,j)*lLLL(i, j, 3));
                    lFr_x(i, j, 3) = lrhol(i,j)/(2*c_Gamma)*((c_Gamma-1)*(lul(i,j)*lul(i,j)-lvl(i,j)*lvl(i,j))*lLLL(i, j, 0)+2*(c_Gamma-1)*lvl(i,j)*lvl(i,j)*lLLL(i, j, 1)
                                    +(lhl(i,j)-lal(i,j)*lul(i,j))*lLLL(i, j, 2)+(lhl(i,j)+lal(i,j)*lul(i,j))*lLLL(i, j, 3));

                    // ------------------------------------------------------------------
                    lLLL(i, j, 0) = lul(i,j);
                    lLLL(i, j, 1) = lul(i,j);
                    lLLL(i, j, 2) = lul(i,j) - lal(i,j);
                    lLLL(i, j, 3) = lul(i,j) + lal(i,j);

                    for(int n = 0; n < c_Nvar; n++)
                    {                                                                          
                        lLLL(i, j, n) = 0.5*(lLLL(i, j, n) - sqrt( fabs(Kokkos::pow(lLLL(i, j, n),2) + Kokkos::pow(0.0001,2)) ) );
                    }

                    lFl_x(i, j, 0) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lLLL(i, j, 0)+lLLL(i, j, 2)+lLLL(i, j, 3));
                    lFl_x(i, j, 1) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lul(i,j)*lLLL(i, j, 0)+(lul(i,j)-lal(i,j))*lLLL(i, j, 2)+(lul(i,j)+lal(i,j))*lLLL(i, j, 3));
                    lFl_x(i, j, 2) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lvl(i,j)*lLLL(i, j, 1)+lvl(i,j)*lLLL(i, j, 2)+lvl(i,j)*lLLL(i, j, 3));
                    lFl_x(i, j, 3) = lrhol(i,j)/(2*c_Gamma)*((c_Gamma-1)*(lul(i,j)*lul(i,j)-lvl(i,j)*lvl(i,j))*lLLL(i, j, 0)+2*(c_Gamma-1)*lvl(i,j)*lvl(i,j)*lLLL(i, j, 1)
                                    +(lhl(i,j)-lal(i,j)*lul(i,j))*lLLL(i, j, 2)+(lhl(i,j)+lal(i,j)*lul(i,j))*lLLL(i, j, 3));
            });  
    Kokkos::fence();

// // ============================================================== test
//     auto Fr_mirror = Kokkos::create_mirror_view(Fr_x);
//     Kokkos::deep_copy(Fr_mirror, Fr_x);
//     auto dim0 = Fr_x.extent(0); // 第一维度的大小
    
//     // 遍历Fr_mirror来输出数据
// for(size_t i = 3; i < (dim0/20); ++i) {
        
//             std::cout << "Fr_x(" << i <<  ") = " << Fr_mirror(i,10,1) << std::endl;
//     }   
// // ============================================================== test

    // Kokkos::fence();    

// - WENO5
    Kokkos::parallel_for("WENO5", Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<3>>({3, 3, 0}, {c_Nx+4, c_Ny+4, c_Nvar}), 
            KOKKOS_LAMBDA (const int i, const int j, const int n) 
            {
                
                    lHp1(i,j,n) = Rbeta[0]*lFr_x(i-2,j,n)+Rbeta[1]*lFr_x(i-1,j,n)+Rbeta[2]*lFr_x(i,j,n);
                    lHp2(i,j,n) = Rbeta[3]*lFr_x(i-1,j,n)+Rbeta[4]*lFr_x(i,j,n)+Rbeta[5]*lFr_x(i+1,j,n);
                    lHp3(i,j,n) = Rbeta[6]*lFr_x(i,j,n)+Rbeta[7]*lFr_x(i+1,j,n)+Rbeta[8]*lFr_x(i+2,j,n);
                    lHn1(i,j,n) = Lbeta[0]*lFl_x(i-1,j,n)+Lbeta[1]*lFl_x(i,j,n)+Lbeta[2]*lFl_x(i+1,j,n);
                    lHn2(i,j,n) = Lbeta[3]*lFl_x(i,j,n)+Lbeta[4]*lFl_x(i+1,j,n)+Lbeta[5]*lFl_x(i+2,j,n);
                    lHn3(i,j,n) = Lbeta[6]*lFl_x(i+1,j,n)+Lbeta[7]*lFl_x(i+2,j,n)+Lbeta[8]*lFl_x(i+3,j,n);

                    lISr1(i,j,n)=IS1* Kokkos::pow((lFr_x(i-2,j,n)-2*lFr_x(i-1,j,n)+lFr_x(i,j,n)),2)+IS2* Kokkos::pow((lFr_x(i-2,j,n)-4*lFr_x(i-1,j,n)+3*lFr_x(i,j,n)),2);
                    lISr2(i,j,n)=IS1* Kokkos::pow((lFr_x(i-1,j,n)-2*lFr_x(i,j,n)+lFr_x(i+1,j,n)),2)+IS2* Kokkos::pow((lFr_x(i-1,j,n)-lFr_x(i+1,j,n)),2);
                    lISr3(i,j,n)=IS1* Kokkos::pow((lFr_x(i,j,n)-2*lFr_x(i+1,j,n)+lFr_x(i+2,j,n)),2)+IS2* Kokkos::pow((3*lFr_x(i,j,n)-4*lFr_x(i+1,j,n)+lFr_x(i+2,j,n)),2);
                    lISl1(i,j,n)=IS1* Kokkos::pow((lFl_x(i-1,j,n)-2*lFl_x(i,j,n)+lFl_x(i+1,j,n)),2)+IS2* Kokkos::pow((lFl_x(i-1,j,n)-4*lFl_x(i,j,n)+3*lFl_x(i+1,j,n)),2);
                    lISl2(i,j,n)=IS1* Kokkos::pow((lFl_x(i,j,n)-2*lFl_x(i+1,j,n)+lFl_x(i+2,j,n)),2)+IS2* Kokkos::pow((lFl_x(i+2,j,n)-lFl_x(i,j,n)),2);
                    lISl3(i,j,n)=IS1* Kokkos::pow((lFl_x(i+1,j,n)-2*lFl_x(i+2,j,n)+lFl_x(i+3,j,n)),2)+IS2* Kokkos::pow((3*lFl_x(i+1,j,n)-4*lFl_x(i+2,j,n)+lFl_x(i+3,j,n)),2);

                    lalphr1(i,j,n)=Cr[0]/ Kokkos::pow((Epsilon+lISr1(i,j,n)),2);
                    lalphr2(i,j,n)=Cr[1]/ Kokkos::pow((Epsilon+lISr2(i,j,n)),2);
                    lalphr3(i,j,n)=Cr[2]/ Kokkos::pow((Epsilon+lISr3(i,j,n)),2);
                    lalphl1(i,j,n)=Cl[0]/ Kokkos::pow((Epsilon+lISl1(i,j,n)),2);
                    lalphl2(i,j,n)=Cl[1]/ Kokkos::pow((Epsilon+lISl2(i,j,n)),2);
                    lalphl3(i,j,n)=Cl[2]/ Kokkos::pow((Epsilon+lISl3(i,j,n)),2);

                    lomegar1(i,j,n)=lalphr1(i,j,n)/(lalphr1(i,j,n)+lalphr2(i,j,n)+lalphr3(i,j,n));
                    lomegar2(i,j,n)=lalphr2(i,j,n)/(lalphr1(i,j,n)+lalphr2(i,j,n)+lalphr3(i,j,n));
                    lomegar3(i,j,n)=lalphr3(i,j,n)/(lalphr1(i,j,n)+lalphr2(i,j,n)+lalphr3(i,j,n));
                    lomegal1(i,j,n)=lalphl1(i,j,n)/(lalphl1(i,j,n)+lalphl2(i,j,n)+lalphl3(i,j,n));
                    lomegal2(i,j,n)=lalphl2(i,j,n)/(lalphl1(i,j,n)+lalphl2(i,j,n)+lalphl3(i,j,n));
                    lomegal3(i,j,n)=lalphl3(i,j,n)/(lalphl1(i,j,n)+lalphl2(i,j,n)+lalphl3(i,j,n));
                    
                    lGr(i,j,n)=lomegar1(i,j,n)*lHp1(i,j,n)+lomegar2(i,j,n)*lHp2(i,j,n)+lomegar3(i,j,n)*lHp3(i,j,n);
                    lGl(i,j,n)=lomegal1(i,j,n)*lHn1(i,j,n)+lomegal2(i,j,n)*lHn2(i,j,n)+lomegal3(i,j,n)*lHn3(i,j,n);
                    
            });  
    Kokkos::fence();   

    Kokkos::parallel_for("WENO5", Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<3>>({3, 3, 0}, {c_Nx+4, c_Ny+4, c_Nvar}), 
            KOKKOS_LAMBDA (const int i, const int j, const int n) 
            {
                RHS(i,j,n) = -(lGr(i,j,n)+lGl(i,j,n)-lGr(i-1,j,n)-lGl(i-1,j,n))/dx;
            });  

    Kokkos::fence();
    // ============================================================== test
//     auto RHS_X_mirror = Kokkos::create_mirror_view(RHS);
//     Kokkos::deep_copy(RHS_X_mirror, RHS);
//     auto dim0 = RHS.extent(0); // 第一维度的大小
    
//     // 遍历Fr_mirror来输出数据
// for(size_t i = 3; i < (dim0/20); ++i) {
        
//             std::cout << "RHS_x(" << i <<  ") = " << RHS_X_mirror(i,10,1) << std::endl;
//     }   
// ============================================================== test
}

//-----------------------------------------------------------------------------------------------
void InviscidFluxCalculator::InviscidFlux_y( view_type &U, view_type &RHS){
    using flcl::view_from_ndarray;

    double dy = Ly/c_Ny;

    // int Direction = 2;
    double Epsilon = 1e-6;
    double IS1, IS2;
    double Cl[3], Cr[3];
    double Rbeta[9], Lbeta[9];
    
    // Initialize value
    IS1 = 13.0/12.0;        IS2 = 1.0/4.0;
    Cl[0] = 3.0/10.0;       Cr[0] = 1.0/10.0;
    Cl[1] = 3.0/5.0;        Cr[1] = 3.0/5.0;
    Cl[2] = 1.0/10.0;       Cr[2] = 3.0/10.0;
    Rbeta[0] =  1.0/3.0;   Lbeta[0] = -1.0/6.0;
    Rbeta[1] = -7.0/6.0;   Lbeta[1] =  5.0/6.0;
    Rbeta[2] =  11.0/6.0;  Lbeta[2] =  1.0/3.0;
    Rbeta[3] = -1.0/6.0;   Lbeta[3] =  1.0/3.0;
    Rbeta[4] =  5.0/6.0;   Lbeta[4] =  5.0/6.0;
    Rbeta[5] =  1.0/3.0;   Lbeta[5] = -1.0/6.0;
    Rbeta[6] =  1.0/3.0;   Lbeta[6] =  11.0/6.0;
    Rbeta[7] =  5.0/6.0;   Lbeta[7] = -7.0/6.0;
    Rbeta[8] = -1.0/6.0;   Lbeta[8] =  1.0/3.0;

    auto lLLL = LLL;
    auto lFr_y = Fr_y; auto lFl_y = Fl_y;

    auto lGl = Gl; auto lGr = Gr;
    auto lHp1= Hp1; auto lHn1 = Hn1;
    auto lHp2 = Hp2; auto lHn2 = Hn2;
    auto lHp3 = Hp3; auto lHn3 = Hn3;
    auto lISl1 = ISl1; auto lISr1 = ISr1;
    auto lISl2 = ISl2; auto lISr2 = ISr2;
    auto lISl3 = ISl3; auto lISr3 = ISr3;
    auto lomegal1 = omegal1; auto lomegar1 = omegar1;
    auto lomegal2 = omegal2; auto lomegar2 = omegar2;
    auto lomegal3 = omegal3; auto lomegar3 = omegar3;
    auto lalphl1 = alphl1;  auto lalphr1 = alphr1;
    auto lalphl2 = alphl2;  auto lalphr2 = alphr2;
    auto lalphl3 = alphl3;  auto lalphr3 = alphr3;

    auto lrhol = rhol;  auto lal = al;
    auto lul = ul;  auto lvl = vl;
    auto lhl = hl;  auto lpl = pl;
    
//- SW split
    Kokkos::parallel_for("SW split", Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<2>>({0, 0}, {c_Nx+7, c_Ny+7}), 
            KOKKOS_LAMBDA (const int i, const int j) 
            {

                    lrhol(i,j) = U(i,j,0);
                    lul(i,j) = U(i,j,1)/lrhol(i,j);
                    lvl(i,j) = U(i,j,2)/lrhol(i,j);
                    lpl(i,j) = (c_Gamma-1)*(U(i,j,3)-0.5*lrhol(i,j)*(lul(i,j)*lul(i,j)+lvl(i,j)*lvl(i,j)));
                    lhl(i,j) = (U(i,j,3)+lpl(i,j))/lrhol(i,j);
                    if(lpl(i,j) < 0.0)
                    {
                        lpl(i,j)=1.e-4;
                    }
                    lal(i,j) = sqrt(fabs(c_Gamma*lpl(i,j)/lrhol(i,j)));
                    
                    lLLL(i, j, 0) = lvl(i,j);
                    lLLL(i, j, 1) = lvl(i,j);
                    lLLL(i, j, 2) = lvl(i,j) - lal(i,j);
                    lLLL(i, j, 3) = lvl(i,j) + lal(i,j);
                                                                                               
                    for(int n = 0; n < c_Nvar; n++)
                    {                                                                          
                        lLLL(i, j, n) = 0.5*(lLLL(i, j, n) + sqrt( fabs(Kokkos::pow(lLLL(i, j, n),2) + Kokkos::pow(0.0001,2)) ) );
                    }

                    lFr_y(i, j, 0) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lLLL(i, j, 1)+lLLL(i, j, 2)+lLLL(i, j, 3));
                    lFr_y(i, j, 1) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lul(i,j)*lLLL(i, j, 0)+lul(i,j)*lLLL(i, j, 2)+lul(i,j)*lLLL(i, j, 3));
                    lFr_y(i, j, 2) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lvl(i,j)*lLLL(i, j, 1)+(lvl(i,j)-lal(i,j))*lLLL(i, j, 2)+(lvl(i,j)+lal(i,j))*lLLL(i, j, 3));
                    lFr_y(i, j, 3) = lrhol(i,j)/(2*c_Gamma)*((c_Gamma-1)*(lvl(i,j)*lvl(i,j)-lul(i,j)*lul(i,j))*lLLL(i, j, 1)+2*(c_Gamma-1)*lul(i,j)*lul(i,j)*lLLL(i, j, 0)
                                +(lhl(i,j)-lal(i,j)*lvl(i,j))*lLLL(i, j, 2)+(lhl(i,j)+lal(i,j)*lvl(i,j))*lLLL(i, j, 3));


                                            // lLLL(i,j,0) = 0.5*(lvl(i,j) + sqrt(fabs(lvl(i,j)*lvl(i,j)+0.0001*0.0001)) );
                                            // lLLL(i,j,1) = 0.5*(lvl(i,j) - sqrt(fabs(lvl(i,j)*lvl(i,j)+0.0001*0.0001)) );
                                            // lLLL(i,j,2) = 0.5*(lvl(i,j)-lal(i,j)) + sqrt(fabs((lvl(i,j)-lal(i,j))*(lvl(i,j)-lal(i,j))+0.0001*0.0001));
                                            // lLLL(i,j,3) = 0.5*(lvl(i,j)-lal(i,j)) - sqrt(fabs((lvl(i,j)-lal(i,j))*(lvl(i,j)-lal(i,j))+0.0001*0.0001));
                                            // lLLL(i,j,4) = 0.5*(lvl(i,j)+lal(i,j)) + sqrt(fabs((lvl(i,j)+lal(i,j))*(lvl(i,j)+lal(i,j))+0.0001*0.0001));
                                            // lLLL(i,j,5) = 0.5*(lvl(i,j)+lal(i,j)) - sqrt(fabs((lvl(i,j)+lal(i,j))*(lvl(i,j)+lal(i,j))+0.0001*0.0001));

                                            // lFr_y(i, j, 0) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lLLL(i,j,0)+lLLL(i,j,2)+lLLL(i,j,4));
                                            // lFr_y(i, j, 1) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lul(i,j)*lLLL(i, j, 0)+lul(i,j)*lLLL(i, j, 2)+lul(i,j)*lLLL(i, j, 4));
                                            // lFr_y(i, j, 2) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lvl(i,j)*lLLL(i, j, 0)+(lvl(i,j)-lal(i,j))*lLLL(i, j, 2)+(lvl(i,j)+lal(i,j))*lLLL(i, j, 4));
                                            // lFr_y(i, j, 3) = lrhol(i,j)/(2*c_Gamma)* ((c_Gamma-1)*(lvl(i,j)*lvl(i,j)+lul(i,j)*lul(i,j))*lLLL(i, j, 0)+(lhl(i,j)-lvl(i,j)*lal(i,j))*lLLL(i,j,2)+(lhl(i,j)+lvl(i,j)*lal(i,j))*lLLL(i,j,4));

                                            // lFl_y(i, j, 0) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lLLL(i, j, 1)+lLLL(i, j, 3)+lLLL(i, j, 5));
                                            // lFl_y(i, j, 1) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lul(i,j)*lLLL(i, j, 1)+lul(i,j)*lLLL(i, j, 3)+lul(i,j)*lLLL(i, j, 5));
                                            // lFl_y(i, j, 2) = lrhol(i,j)/(2*c_Gamma)* (2*(c_Gamma-1)*lvl(i,j)*lLLL(i, j, 1)+(lvl(i,j)-lal(i,j))*lLLL(i, j, 3)+(lvl(i,j)+lal(i,j))*lLLL(i, j, 5));
                                            // lFl_y(i, j, 3) = lrhol(i,j)/(2*c_Gamma)* ((c_Gamma-1)*(lvl(i,j)*lvl(i,j)+lul(i,j)*lul(i,j))*lLLL(i, j, 1)+(lhl(i,j)-lvl(i,j)*lal(i,j))*lLLL(i,j,3)+(lhl(i,j)+lvl(i,j)*lal(i,j))*lLLL(i,j,5));

                    lLLL(i, j, 0) = lvl(i,j);
                    lLLL(i, j, 1) = lvl(i,j);
                    lLLL(i, j, 2) = lvl(i,j) - lal(i,j);
                    lLLL(i, j, 3) = lvl(i,j) + lal(i,j);

                    for(int n = 0; n < c_Nvar; n++)
                    {                                                                          
                        lLLL(i, j, n) = 0.5*(lLLL(i, j, n) - sqrt(fabs(Kokkos::pow(lLLL(i, j, n),2) + Kokkos::pow(0.0001,2)) ) );
                    }
                    lFl_y(i, j, 0) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lLLL(i, j, 1)+lLLL(i, j, 2)+lLLL(i, j, 3));
                    lFl_y(i, j, 1) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lul(i,j)*lLLL(i, j, 0)+lul(i,j)*lLLL(i, j, 2)+lul(i,j)*lLLL(i, j, 3));
                    lFl_y(i, j, 2) = lrhol(i,j)/(2*c_Gamma)*(2*(c_Gamma-1)*lvl(i,j)*lLLL(i, j, 1)+(lvl(i,j)-lal(i,j))*lLLL(i, j, 2)+(lvl(i,j)+lal(i,j))*lLLL(i, j, 3));
                    lFl_y(i, j, 3) = lrhol(i,j)/(2*c_Gamma)*((c_Gamma-1)*(lvl(i,j)*lvl(i,j)-lul(i,j)*lul(i,j))*lLLL(i, j, 1)+2*(c_Gamma-1)*lul(i,j)*lul(i,j)*lLLL(i, j, 0)
                                +(lhl(i,j)-lal(i,j)*lvl(i,j))*lLLL(i, j, 2)+(lhl(i,j)+lal(i,j)*lvl(i,j))*lLLL(i, j, 3));   
            });      
    Kokkos::fence();

// // ============================================================== test
//     auto rhol_mirror = Kokkos::create_mirror_view(rhol);
//     Kokkos::deep_copy(rhol_mirror, rhol);
//     auto dim0 = rhol.extent(0); // 第一维度的大小
    
//     // 遍历Fr_mirror来输出数据
// for(size_t i = 2; i < 10; ++i) {
        
//             std::cout << "rhol(" << i <<  ") = " << rhol_mirror(i,50) << std::endl;
//     }   
// // ============================================================== test

// - WENO5
    Kokkos::parallel_for("WENO5-H", Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<3>>({3, 3, 0}, {c_Nx+4, c_Ny+4, c_Nvar}), 
            KOKKOS_LAMBDA (const int i, const int j, const int n) 
            {
                    lHp1(i,j,n) = Rbeta[0]*lFr_y(i,j-2,n)+Rbeta[1]*lFr_y(i,j-1,n)+Rbeta[2]*lFr_y(i,j,n);
                    lHp2(i,j,n) = Rbeta[3]*lFr_y(i,j-1,n)+Rbeta[4]*lFr_y(i,j,n)+Rbeta[5]*lFr_y(i,j+1,n);
                    lHp3(i,j,n) = Rbeta[6]*lFr_y(i,j,n)+Rbeta[7]*lFr_y(i,j+1,n)+Rbeta[8]*lFr_y(i,j+2,n);
                    lHn1(i,j,n) = Lbeta[0]*lFl_y(i,j-1,n)+Lbeta[1]*lFl_y(i,j,n)+Lbeta[2]*lFl_y(i,j+1,n);
                    lHn2(i,j,n) = Lbeta[3]*lFl_y(i,j,n)+Lbeta[4]*lFl_y(i,j+1,n)+Lbeta[5]*lFl_y(i,j+2,n);
                    lHn3(i,j,n) = Lbeta[6]*lFl_y(i,j+1,n)+Lbeta[7]*lFl_y(i,j+2,n)+Lbeta[8]*lFl_y(i,j+3,n);
                
                    lISr1(i,j,n)=IS1* Kokkos::pow((lFr_y(i,j-2,n)-2*lFr_y(i,j-1,n)+lFr_y(i,j,n)),2)+IS2* Kokkos::pow((lFr_y(i,j-2,n)-4*lFr_y(i,j-1,n)+3*lFr_y(i,j,n)),2);
                    lISr2(i,j,n)=IS1* Kokkos::pow((lFr_y(i,j-1,n)-2*lFr_y(i,j,n)+lFr_y(i,j+1,n)),2)+IS2* Kokkos::pow((lFr_y(i,j-1,n)-lFr_y(i,j+1,n)),2);
                    lISr3(i,j,n)=IS1* Kokkos::pow((lFr_y(i,j,n)-2*lFr_y(i,j+1,n)+lFr_y(i,j+2,n)),2)+IS2* Kokkos::pow((3*lFr_y(i,j,n)-4*lFr_y(i,j+1,n)+lFr_y(i,j+2,n)),2);
                    lISl1(i,j,n)=IS1* Kokkos::pow((lFl_y(i,j-1,n)-2*lFl_y(i,j,n)+lFl_y(i,j+1,n)),2)+IS2* Kokkos::pow((lFl_y(i,j-1,n)-4*lFl_y(i,j,n)+3*lFl_y(i,j+1,n)),2);
                    lISl2(i,j,n)=IS1* Kokkos::pow((lFl_y(i,j,n)-2*lFl_y(i,j+1,n)+lFl_y(i,j+2,n)),2)+IS2* Kokkos::pow((lFl_y(i,j+2,n)-lFl_y(i,j,n)),2);
                    lISl3(i,j,n)=IS1* Kokkos::pow((lFl_y(i,j+1,n)-2*lFl_y(i,j+2,n)+lFl_y(i,j+3,n)),2)+IS2* Kokkos::pow((3*lFl_y(i,j+1,n)-4*lFl_y(i,j+2,n)+lFl_y(i,j+3,n)),2);
                    
                    lalphr1(i,j,n)=Cr[0]/ Kokkos::pow((Epsilon+lISr1(i,j,n)),3);
                    lalphr2(i,j,n)=Cr[1]/ Kokkos::pow((Epsilon+lISr2(i,j,n)),3);
                    lalphr3(i,j,n)=Cr[2]/ Kokkos::pow((Epsilon+lISr3(i,j,n)),3);
                    lalphl1(i,j,n)=Cl[0]/ Kokkos::pow((Epsilon+lISl1(i,j,n)),3);
                    lalphl2(i,j,n)=Cl[1]/ Kokkos::pow((Epsilon+lISl2(i,j,n)),3);
                    lalphl3(i,j,n)=Cl[2]/ Kokkos::pow((Epsilon+lISl3(i,j,n)),3);
                
                    lomegar1(i,j,n)=lalphr1(i,j,n)/(lalphr1(i,j,n)+lalphr2(i,j,n)+lalphr3(i,j,n));
                    lomegar2(i,j,n)=lalphr2(i,j,n)/(lalphr1(i,j,n)+lalphr2(i,j,n)+lalphr3(i,j,n));
                    lomegar3(i,j,n)=lalphr3(i,j,n)/(lalphr1(i,j,n)+lalphr2(i,j,n)+lalphr3(i,j,n));
                    lomegal1(i,j,n)=lalphl1(i,j,n)/(lalphl1(i,j,n)+lalphl2(i,j,n)+lalphl3(i,j,n));
                    lomegal2(i,j,n)=lalphl2(i,j,n)/(lalphl1(i,j,n)+lalphl2(i,j,n)+lalphl3(i,j,n));
                    lomegal3(i,j,n)=lalphl3(i,j,n)/(lalphl1(i,j,n)+lalphl2(i,j,n)+lalphl3(i,j,n));
               
                    lGr(i,j,n)=lomegar1(i,j,n)*lHp1(i,j,n)+lomegar2(i,j,n)*lHp2(i,j,n)+lomegar3(i,j,n)*lHp3(i,j,n);
                    lGl(i,j,n)=lomegal1(i,j,n)*lHn1(i,j,n)+lomegal2(i,j,n)*lHn2(i,j,n)+lomegal3(i,j,n)*lHn3(i,j,n);               
            });
    Kokkos::fence(); 

    Kokkos::parallel_for("WENO5-H", Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<3>>({3, 3, 0}, {c_Nx+4, c_Ny+4, c_Nvar}), 
            KOKKOS_LAMBDA (const int i, const int j, const int n) 
            {
                 RHS(i,j,n) =  RHS(i,j,n) -  (lGr(i,j,n)+lGl(i,j,n)-lGr(i,j-1,n)-lGl(i,j-1,n)) /dy;                
            });

    Kokkos::fence();
// ============================================================== test
//     auto RHS_Y_mirror = Kokkos::create_mirror_view(RHS);
//     Kokkos::deep_copy(RHS_Y_mirror, RHS);
//     auto dim0 = RHS.extent(0); // 第一维度的大小
    
//     // 遍历Fr_mirror来输出数据
// for(size_t i = 1; i < (dim0/20); ++i) {
//             std::cout << "RHS_Y(" << i <<  ") = " << RHS_Y_mirror(i,4,2) << std::endl;
//     }   
// ============================================================== test      
}