// ---------------------------------------------------------------------
//
// Copyright (C) 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

// construct a simplex quadrature, and check that we can get an affine
// transformation out of it.
#ifndef tests_base_simplex_h
#define tests_base_simplex_h

#include "../tests.h"

#include <deal.II/base/quadrature_lib.h>

#include "simplex.h"

#include <numeric>


// Helper functions
template <int dim>
std::array<Point<dim>, dim + 1>
get_simplex();

template <>
std::array<Point<1>, 2>
get_simplex()
{
  return {{Point<1>(3), Point<1>(5)}};
}


template <>
std::array<Point<2>, 3>
get_simplex()
{
  return {{Point<2>(4, 2), Point<2>(3, 3), Point<2>(2, 2.5)}};
}


template <>
std::array<Point<3>, 4>
get_simplex()
{
  return {{Point<3>(4, 2, 0),
           Point<3>(3, 3, 0),
           Point<3>(2, 2.5, 0),
           Point<3>(4.5, 3, 2)}};
}


// Exact integral of 1/R times a polynomial computed using Maple.
double
exact_integral_one_over_r(const unsigned int vertex_index,
                          const unsigned int i,
                          const unsigned int j)
{
  Assert(vertex_index < 4, ExcInternalError());
  Assert(i < 6, ExcNotImplemented());
  Assert(j < 6, ExcNotImplemented());

  // The integrals are computed using the following maple snippet of
  // code:
  //
  //  sing_int := proc(index, N, M)
  //     if index = 0 then
  //        return int(int(x^N *y^M/sqrt(x^2+y^2), x=0.0..1.0), y=0.0..1.0);
  //     elif index = 1 then
  //        return int(int(x^N *y^M/sqrt((x-1)^2+y^2), x=0.0..1.0), y=0.0..1.0);
  //     elif index = 2 then
  //        return int(int(x^N *y^M/sqrt(x^2+(y-1)^2), x=0.0..1.0), y=0.0..1.0);
  //     elif index = 3 then
  //        return int(int((1-x)^N *(1-y)^M/sqrt(x^2+y^2), x=0.0..1.0), y=0.0..1.0);
  //     end if;
  //  end proc;
  //  Digits := 20;
  //  for i from 3 to 3 do
  //     for n from 0 to 5 do
  //      for m from 0 to 5 do
  //           C( v[i+1][n+1][m+1] = sing_int(i, n, m), resultname="a");
  //        end do;
  //     end do;
  //  end do;

  static double v[4][6][6] = {{{0}}};
  if(v[0][0][0] == 0)
    {
      v[0][0][0] = 0.17627471740390860505e1;
      v[0][0][1] = 0.64779357469631903702e0;
      v[0][0][2] = 0.38259785823210634567e0;
      v[0][0][3] = 0.26915893322379450224e0;
      v[0][0][4] = 0.20702239737104695572e0;
      v[0][0][5] = 0.16800109713227567467e0;
      v[0][1][0] = 0.64779357469631903702e0;
      v[0][1][1] = 0.27614237491539669920e0;
      v[0][1][2] = 0.17015838751246776515e0;
      v[0][1][3] = 0.12189514164974600651e0;
      v[0][1][4] = 0.94658660368131133694e-1;
      v[0][1][5] = 0.77263794021029438797e-1;
      v[0][2][0] = 0.38259785823210634567e0;
      v[0][2][1] = 0.17015838751246776515e0;
      v[0][2][2] = 0.10656799507071040471e0;
      v[0][2][3] = 0.76947022258735165920e-1;
      v[0][2][4] = 0.60022626787495395021e-1;
      v[0][2][5] = 0.49131622931360879320e-1;
      v[0][3][0] = 0.26915893322379450224e0;
      v[0][3][1] = 0.12189514164974600651e0;
      v[0][3][2] = 0.76947022258735165919e-1;
      v[0][3][3] = 0.55789184535895709637e-1;
      v[0][3][4] = 0.43625068213915842136e-1;
      v[0][3][5] = 0.35766126849971778500e-1;
      v[0][4][0] = 0.20702239737104695572e0;
      v[0][4][1] = 0.94658660368131133694e-1;
      v[0][4][2] = 0.60022626787495395021e-1;
      v[0][4][3] = 0.43625068213915842137e-1;
      v[0][4][4] = 0.34164088852375945192e-1;
      v[0][4][5] = 0.28037139560980277614e-1;
      v[0][5][0] = 0.16800109713227567467e0;
      v[0][5][1] = 0.77263794021029438797e-1;
      v[0][5][2] = 0.49131622931360879320e-1;
      v[0][5][3] = 0.35766126849971778501e-1;
      v[0][5][4] = 0.28037139560980277614e-1;
      v[0][5][5] = 0.23024181049838367777e-1;
      v[1][0][0] = 0.17627471740390860505e1;
      v[1][0][1] = 0.64779357469631903702e0;
      v[1][0][2] = 0.38259785823210634567e0;
      v[1][0][3] = 0.26915893322379450224e0;
      v[1][0][4] = 0.20702239737104695572e0;
      v[1][0][5] = 0.16800109713227567467e0;
      v[1][1][0] = 0.11149535993427670134e1;
      v[1][1][1] = 0.37165119978092233782e0;
      v[1][1][2] = 0.21243947071963858053e0;
      v[1][1][3] = 0.14726379157404849573e0;
      v[1][1][4] = 0.11236373700291582202e0;
      v[1][1][5] = 0.90737303111246235871e-1;
      v[1][2][0] = 0.84975788287855432210e0;
      v[1][2][1] = 0.26566721237799340376e0;
      v[1][2][2] = 0.14884907827788122009e0;
      v[1][2][3] = 0.10231567218303765515e0;
      v[1][2][4] = 0.77727703422280083352e-1;
      v[1][2][5] = 0.62605132021577676395e-1;
      v[1][3][0] = 0.69800109142265347423e0;
      v[1][3][1] = 0.20794647083778622837e0;
      v[1][3][2] = 0.11487965864809909847e0;
      v[1][3][3] = 0.78525390514866270852e-1;
      v[1][3][4] = 0.59489228415223897572e-1;
      v[1][3][5] = 0.47838457013298217744e-1;
      v[1][4][0] = 0.59754668912231692323e0;
      v[1][4][1] = 0.17125249387868593878e0;
      v[1][4][2] = 0.93606816359052444729e-1;
      v[1][4][3] = 0.63728830247554475330e-1;
      v[1][4][4] = 0.48187332620207367724e-1;
      v[1][4][5] = 0.38708290797416359020e-1;
      v[1][5][0] = 0.52527944036356840363e0;
      v[1][5][1] = 0.14574366656617935708e0;
      v[1][5][2] = 0.78997159795636003667e-1;
      v[1][5][3] = 0.53620816423066464705e-1;
      v[1][5][4] = 0.40487985967086264433e-1;
      v[1][5][5] = 0.32498604596082509165e-1;
      v[2][0][0] = 0.17627471740390860505e1;
      v[2][0][1] = 0.11149535993427670134e1;
      v[2][0][2] = 0.84975788287855432210e0;
      v[2][0][3] = 0.69800109142265347419e0;
      v[2][0][4] = 0.59754668912231692318e0;
      v[2][0][5] = 0.52527944036356840362e0;
      v[2][1][0] = 0.64779357469631903702e0;
      v[2][1][1] = 0.37165119978092233782e0;
      v[2][1][2] = 0.26566721237799340376e0;
      v[2][1][3] = 0.20794647083778622835e0;
      v[2][1][4] = 0.17125249387868593876e0;
      v[2][1][5] = 0.14574366656617935708e0;
      v[2][2][0] = 0.38259785823210634567e0;
      v[2][2][1] = 0.21243947071963858053e0;
      v[2][2][2] = 0.14884907827788122009e0;
      v[2][2][3] = 0.11487965864809909845e0;
      v[2][2][4] = 0.93606816359052444712e-1;
      v[2][2][5] = 0.78997159795636003667e-1;
      v[2][3][0] = 0.26915893322379450223e0;
      v[2][3][1] = 0.14726379157404849572e0;
      v[2][3][2] = 0.10231567218303765514e0;
      v[2][3][3] = 0.78525390514866270835e-1;
      v[2][3][4] = 0.63728830247554475311e-1;
      v[2][3][5] = 0.53620816423066464702e-1;
      v[2][4][0] = 0.20702239737104695572e0;
      v[2][4][1] = 0.11236373700291582202e0;
      v[2][4][2] = 0.77727703422280083352e-1;
      v[2][4][3] = 0.59489228415223897563e-1;
      v[2][4][4] = 0.48187332620207367713e-1;
      v[2][4][5] = 0.40487985967086264434e-1;
      v[2][5][0] = 0.16800109713227567468e0;
      v[2][5][1] = 0.90737303111246235879e-1;
      v[2][5][2] = 0.62605132021577676399e-1;
      v[2][5][3] = 0.47838457013298217740e-1;
      v[2][5][4] = 0.38708290797416359014e-1;
      v[2][5][5] = 0.32498604596082509169e-1;
      v[3][0][0] = 0.17627471740390860505e1;
      v[3][0][1] = 0.11149535993427670134e1;
      v[3][0][2] = 0.84975788287855432210e0;
      v[3][0][3] = 0.69800109142265347419e0;
      v[3][0][4] = 0.59754668912231692318e0;
      v[3][0][5] = 0.52527944036356840362e0;
      v[3][1][0] = 0.11149535993427670134e1;
      v[3][1][1] = 0.74330239956184467563e0;
      v[3][1][2] = 0.58409067050056091834e0;
      v[3][1][3] = 0.49005462058486724584e0;
      v[3][1][4] = 0.42629419524363098443e0;
      v[3][1][5] = 0.37953577379738904654e0;
      v[3][2][0] = 0.84975788287855432210e0;
      v[3][2][1] = 0.58409067050056091834e0;
      v[3][2][2] = 0.46727253640044873467e0;
      v[3][2][3] = 0.39698780839518011595e0;
      v[3][2][4] = 0.34864851772399749038e0;
      v[3][2][5] = 0.31278926702684569312e0;
      v[3][3][0] = 0.69800109142265347423e0;
      v[3][3][1] = 0.49005462058486724586e0;
      v[3][3][2] = 0.39698780839518011599e0;
      v[3][3][3] = 0.34027526433872581371e0;
      v[3][3][4] = 0.30088082631586196583e0;
      v[3][3][5] = 0.27141910362887187844e0;
      v[3][4][0] = 0.59754668912231692323e0;
      v[3][4][1] = 0.42629419524363098445e0;
      v[3][4][2] = 0.34864851772399749044e0;
      v[3][4][3] = 0.30088082631586196576e0;
      v[3][4][4] = 0.26744962339187730308e0;
      v[3][4][5] = 0.24229245314748740295e0;
      v[3][5][0] = 0.52527944036356840363e0;
      v[3][5][1] = 0.37953577379738904655e0;
      v[3][5][2] = 0.31278926702684569301e0;
      v[3][5][3] = 0.27141910362887187862e0;
      v[3][5][4] = 0.24229245314748740263e0;
      v[3][5][5] = 0.22026586649771582089e0;
    }
  return v[vertex_index][i][j];
}



double
exact_integral_one_over_r_middle(const unsigned int i, const unsigned int j)
{
  Assert(i < 6, ExcNotImplemented());
  Assert(j < 6, ExcNotImplemented());

  // The integrals are computed using the following Mathematica snippet of
  // code:
  //
  // x0 = 0.5
  // y0 = 0.5
  // Do[Do[Print["v[", n, "][", m, "]=",
  //    NumberForm[
  //     NIntegrate[
  //      x^n*y^m/Sqrt[(x - x0)^2 + (y - y0)^2], {x, 0, 1}, {y, 0, 1},
  //      MaxRecursion -> 10000, PrecisionGoal -> 9], 9], ";"], {n, 0,
  //    4}], {m, 0, 4}]


  static double v[6][6] = {{0}};

  if(v[0][0] == 0)
    {
      v[0][0] = 3.52549435;
      ;
      v[1][0] = 1.76274717;
      v[2][0] = 1.07267252;
      v[3][0] = 0.727635187;
      v[4][0] = 0.53316959;
      v[0][1] = 1.76274717;
      v[1][1] = 0.881373587;
      v[2][1] = 0.536336258;
      v[3][1] = 0.363817594;
      v[4][1] = 0.266584795;
      v[0][2] = 1.07267252;
      v[1][2] = 0.536336258;
      v[2][2] = 0.329313861;
      v[3][2] = 0.225802662;
      v[4][2] = 0.167105787;
      v[0][3] = 0.727635187;
      v[1][3] = 0.363817594;
      v[2][3] = 0.225802662;
      v[3][3] = 0.156795196;
      v[4][3] = 0.117366283;
      v[0][4] = 0.53316959;
      v[1][4] = 0.266584795;
      v[2][4] = 0.167105787;
      v[3][4] = 0.117366283;
      v[4][4] = 0.0887410133;
    }
  return v[i][j];
}

#endif
