// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2018 by the deal.II authors
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

// check SparseMatrix::mmult

#include "../tests.h"
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <iostream>
#include <vector>

void
test()
{
  // A = [1, 0, 0; 4, 1, 0; 7, 2, 1]
  PETScWrappers::SparseMatrix A(3, 3, 3);
  A.set(0, 0, 1);
  A.set(1, 0, 4);
  A.set(1, 1, 1);
  A.set(2, 0, 7);
  A.set(2, 1, 2);
  A.set(2, 2, 1);
  A.compress(VectorOperation::insert);

  // B = [1, 2, 3; 0, -3, -6; 0, 0, 0]
  PETScWrappers::SparseMatrix B(3, 3, 3);
  B.set(0, 0, 1);
  B.set(0, 1, 2);
  B.set(0, 2, 3);
  B.set(1, 1, -3);
  B.set(1, 2, -6);
  B.compress(VectorOperation::insert);

  PETScWrappers::SparseMatrix C(3, 3, 3);

  // C := AB = [1, 2, 3; 4, 5, 6; 7, 8, 9]
  A.mmult(C, B);

  // make sure we get the expected result
  for (unsigned int i = 0; i < C.m(); ++i)
    for (unsigned int j = 0; j < C.n(); ++j)
      AssertThrow(C(i, j) == 3 * i + j + 1, ExcInternalError());

  deallog << "OK" << std::endl;
}

int
main(int argc, char** argv)
{
  initlog();

  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      {
        test();
      }
    }
  catch (std::exception& exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };
}
