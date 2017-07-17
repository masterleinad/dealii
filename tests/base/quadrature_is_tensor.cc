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



// check the type_trait is_tensor_product for all the quadrature classes


#include "../tests.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature.h>

template <int dim>
void
print_is_tensor_product()
{
  deallog << "Quadrature<"             << dim << ">: "
          << is_tensor_product<Quadrature<dim>>()             << std::endl;
  deallog << "QIterated<"              << dim << ">: "
          << is_tensor_product<QIterated<dim>>()              << std::endl;
  deallog << "QGauss<"                 << dim << ">: "
          << is_tensor_product<QGauss<dim>>()                 << std::endl;
  deallog << "QGaussLobatto<"          << dim << ">: "
          << is_tensor_product<QGaussLobatto<dim>>()          << std::endl;
  deallog << "QMidpoint<"              << dim << ">: "
          << is_tensor_product<QMidpoint<dim>>()              << std::endl;
  deallog << "QSimpson<"               << dim << ">: "
          << is_tensor_product<QIterated<dim>>()              << std::endl;
  deallog << "QTrapez<"                << dim << ">: "
          << is_tensor_product<QTrapez<dim>>()                << std::endl;
  deallog << "QMilne<"                 << dim << ">: "
          << is_tensor_product<QMilne<dim>>()                 << std::endl;
  deallog << "QWeddle<"                << dim << ">: "
          << is_tensor_product<QWeddle<dim>>()                << std::endl;
  deallog << "QGaussChebyshev<"        << dim << ">: "
          << is_tensor_product<QGaussChebyshev<dim>>()        << std::endl;
  deallog << "QGaussRadauChebyshev<"   << dim << ">: "
          << is_tensor_product<QGaussRadauChebyshev<dim>>()   << std::endl;
  deallog << "QGaussLobattoChebyshev<" << dim << ">: "
          << is_tensor_product<QGaussLobattoChebyshev<dim>>() << std::endl;
  deallog << "QAnisotropic<"           << dim << ">: "
          << is_tensor_product<QAnisotropic<dim>>()           << std::endl;
  deallog << "QGaussLog<"              << dim << ">: "
          << is_tensor_product<QGaussLog<dim>>()              << std::endl;
  deallog << "QGaussLogR<"             << dim << ">: "
          << is_tensor_product<QGaussLogR<dim>>()             << std::endl;
  deallog << "QGaussOneOverR<"         << dim << ">: "
          << is_tensor_product<QGaussOneOverR<dim>>()         << std::endl;
  deallog << "QSorted<"               << dim << ">: "
          << is_tensor_product<QSorted<dim>>()                << std::endl;
  deallog << "QTelles<"               << dim << ">: "
          << is_tensor_product<QTelles<dim>>()                << std::endl;
}

int main()
{
  initlog();
  deallog << std::boolalpha;
  print_is_tensor_product<1>();
  print_is_tensor_product<2>();
  print_is_tensor_product<3>();
}

