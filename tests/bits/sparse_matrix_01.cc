// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2017 by the deal.II authors
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

// check setting elements in a sparse matrix using
// SparseMatrix::set()

#include "../tests.h"
#include <deal.II/lac/sparse_matrix.h>

void
test()
{
  SparsityPattern sp(5, 5, 3);
  for(unsigned int i = 0; i < 5; ++i)
    for(unsigned int j = 0; j < 5; ++j)
      if((i + 2 * j + 1) % 3 == 0)
        sp.add(i, j);
  sp.compress();

  SparseMatrix<double> m(sp);

  // first set a few entries
  for(unsigned int i = 0; i < m.m(); ++i)
    for(unsigned int j = 0; j < m.n(); ++j)
      if((i + 2 * j + 1) % 3 == 0)
        m.set(i, j, i * j * .5 + .5);

  // then make sure we retrieve the same ones
  for(unsigned int i = 0; i < m.m(); ++i)
    for(unsigned int j = 0; j < m.n(); ++j)
      if((i + 2 * j + 1) % 3 == 0)
        {
          AssertThrow(m(i, j) == i * j * .5 + .5, ExcInternalError());
          AssertThrow(m.el(i, j) == i * j * .5 + .5, ExcInternalError());
        }
      else
        {
          AssertThrow(m.el(i, j) == 0, ExcInternalError());
        }

  deallog << "OK" << std::endl;
}

int
main()
{
  initlog();

  try
    {
      test();
    }
  catch(std::exception &exc)
    {
      deallog << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
      deallog << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

      return 1;
    }
  catch(...)
    {
      deallog << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
      deallog << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
      return 1;
    };
}
