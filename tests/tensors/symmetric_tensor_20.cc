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

// check SymmetricTensor<2,dim>::operator= (double)

#include "../tests.h"
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>

int
main()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(3);
  deallog.attach(logfile);

  const unsigned int dim = 3;
  SymmetricTensor<2, dim> t;
  t[0][0] = t[1][0] = t[1][1] = t[2][2] = 3;

  deallog << t.norm() << std::endl;
  t = 0;
  deallog << t.norm() << std::endl;

  Assert(t.norm() == 0, ExcInternalError());

  deallog << "OK" << std::endl;
}
