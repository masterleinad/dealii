// ---------------------------------------------------------------------
//
// Copyright (C) 2006 - 2017 by the deal.II authors
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

// check PointerMatrix:checkVmult

#include "../tests.h"
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/pointer_matrix.h>
#include <deal.II/lac/vector.h>

template <typename number>
void
checkVmult(FullMatrix<number>& A, Vector<number>& V, char* name = "Test Matrix")
{
  deallog << "vmult" << std::endl;

  PointerMatrix<FullMatrix<number>, Vector<number>> P(&A, name);
  Vector<number> O(A.m());
  P.vmult(O, V);

  // Check the dimensions of the result matrix
  Assert(A.m() == O.size(), ExcInternalError());
  deallog << "Dimensions of result vector verified" << std::endl;

  // Verifying results with Method 2: O=A*V
  Vector<number> O_(A.m());
  A.vmult(O_, V);

  Assert(O == O_, ExcInternalError());
  deallog << "Result vector data verified" << std::endl;

  for(unsigned int i = 0; i < O.size(); ++i)
    deallog << O(i) << '\t';
  deallog << std::endl;
}

int
main()
{
  std::ofstream logfile("output");
  deallog << std::fixed;
  deallog << std::setprecision(4);
  deallog.attach(logfile);

  const double Adata[] = {2, 3, 4, 5};

  FullMatrix<double> A(2, 2);
  A.fill(Adata);

  Vector<double> V(2);
  V(0) = 1;
  V(1) = 2;

  checkVmult<double>(A, V);
}
