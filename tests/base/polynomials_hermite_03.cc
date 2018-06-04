// ---------------------------------------------------------------------
//
// Copyright (C) 2018 by the deal.II authors
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

// plot PolynomialsHermite using the constructor specifying degree

#include <deal.II/base/polynomials_hermite.h>

#include "../tests.h"

using namespace std;

void
plot(const Polynomials::Polynomial<double> &poly)
{
  const unsigned int n_points = poly.degree() - 1;
  deallog << "n_points: " << n_points << std::endl;
  std::vector<double> points(n_points);
  for (unsigned int i = 0; i < n_points; ++i)
    {
      std::vector<double> derivatives(2);
      const double        x = i * 1. / (n_points - 1);
      poly.value(x, derivatives);
      deallog << "value at " << x << ": " << derivatives[0]
              << "\t derivative at " << x << ": " << derivatives[1]
              << std::endl;
    }
}

int
main()
{
  initlog();

  for (unsigned int degree = 3; degree < 6; ++degree)
    {
      for (unsigned int i = 0; i < degree + 1; ++i)
        {
          deallog << "Polynomial " << i << std::endl;
          PolynomialsHermite<double> polynomial(degree, i);
          plot(polynomial);
          deallog << std::endl;
        }
    }
}
