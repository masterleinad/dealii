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

// Test that ghosted vectors are read-only

#include "../tests.h"
#include <deal.II/base/index_set.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <iostream>
#include <vector>

void
test()
{
  unsigned int myid    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int numproc = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if(myid == 0)
    deallog << "numproc=" << numproc << std::endl;

  // each processor owns 2 indices and all
  // are ghosting Element 1 (the second)

  IndexSet local_active(numproc * 2);
  local_active.add_range(myid * 2, myid * 2 + 2);
  IndexSet local_relevant(numproc * 2);
  local_relevant.add_range(1, 2);

  PETScWrappers::MPI::Vector vb(local_active, MPI_COMM_WORLD);
  PETScWrappers::MPI::Vector v(local_active, local_relevant, MPI_COMM_WORLD);

  // set local values
  vb(myid * 2)     = myid * 2.0;
  vb(myid * 2 + 1) = myid * 2.0 + 1.0;

  vb.compress(VectorOperation::insert);
  vb *= 2.0;
  v = vb;

  Assert(!vb.has_ghost_elements(), ExcInternalError());
  Assert(v.has_ghost_elements(), ExcInternalError());

  try
    {
      v(0) = 1.0;
    }
  catch(ExceptionBase &e)
    {
      deallog << e.get_exc_name() << std::endl;
    }
  try
    {
      v(0) *= 2.0;
    }
  catch(ExceptionBase &e)
    {
      deallog << e.get_exc_name() << std::endl;
    }
  try
    {
      v += v;
    }
  catch(ExceptionBase &e)
    {
      deallog << e.get_exc_name() << std::endl;
    }

  // done
  if(myid == 0)
    deallog << "OK" << std::endl;
}

int
main(int argc, char **argv)
{
  deal_II_exceptions::disable_abort_on_exception();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  deallog.push(Utilities::int_to_string(myid));

  if(myid == 0)
    {
      initlog();
      deallog << std::setprecision(4);

      test();
    }
  else
    test();
}
