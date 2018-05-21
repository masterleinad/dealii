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

// check Vector<double>::sadd(s, Vector)

#include "../tests.h"
#include <deal.II/lac/vector.h>
#include <vector>

void
test(Vector<double>& v, Vector<double>& w)
{
  for(unsigned int i= 0; i < v.size(); ++i)
    {
      v(i)= i;
      w(i)= i + 1.;
    }

  v.compress();
  w.compress();

  v.sadd(2, w);

  // make sure we get the expected result
  for(unsigned int i= 0; i < v.size(); ++i)
    {
      AssertThrow(w(i) == i + 1., ExcInternalError());
      AssertThrow(v(i) == 2 * i + (i + 1.), ExcInternalError());
    }

  deallog << "OK" << std::endl;
}

int
main()
{
  initlog();

  try
    {
      Vector<double> v(100);
      Vector<double> w(100);
      test(v, w);
    }
  catch(std::exception& exc)
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
