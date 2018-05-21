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

// test Utilities::fixed_power on VectorizedArray, similar to utilities_02

#include "../tests.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

template <int dim>
void
test()
{
  VectorizedArray<double> v1 = make_vectorized_array(2.);
  v1                         = Utilities::fixed_power<dim>(v1);
  deallog << v1[0] << std::endl;
  v1 = -2;
  v1 = Utilities::fixed_power<dim>(v1);
  deallog << v1[0] << std::endl;
  v1 = 2.5;
  v1 = Utilities::fixed_power<dim>(v1);
  deallog << v1[0] << std::endl;
  VectorizedArray<float> v2 = make_vectorized_array<float>(-2.5);
  v2                        = Utilities::fixed_power<dim>(v2);
  deallog << (double)v2[0] << std::endl;
  deallog << std::endl;
}

int
main()
{
  initlog();

  test<0>();
  test<1>();
  test<2>();
  test<3>();
  test<4>();
}
