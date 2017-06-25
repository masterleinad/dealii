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

// the same as timer.cc but also test the new functions last_wall_time(),
// cpu_time() and last_cpu_time().

#include "../tests.h"
#include <deal.II/base/timer.h>
#include <deal.II/base/logstream.h>
#include <fstream>
#include <cmath>
#include <iomanip>

// burn computer time

double s = 0.;
void burn (unsigned int n)
{
  for (unsigned int i=0 ; i<n ; ++i)
    {
      for (unsigned int j=1 ; j<100000 ; ++j)
        {
          s += 1./j * i;
        }
    }
}


int main ()
{
  initlog();
  deallog.threshold_double(1.e-10);

  Timer t;
  burn(50);

  Assert(t.wall_time() > 0., ExcInternalError());
  Assert(t.cpu_time() > 0., ExcInternalError());
  Assert(t.last_wall_time() == 0., ExcInternalError());
  Assert(t.last_cpu_time() == 0, ExcInternalError());

  burn(50);
  t.stop();

  Assert(t.last_wall_time() > 0., ExcInternalError());
  Assert(t.last_cpu_time() > 0, ExcInternalError());

  t.reset();
  Assert(t.wall_time() == 0., ExcInternalError());
  Assert(t.cpu_time()== 0., ExcInternalError());
  Assert(t.last_wall_time() == 0., ExcInternalError());
  Assert(t.last_cpu_time() == 0, ExcInternalError());

  deallog << "OK" << std::endl;
}

