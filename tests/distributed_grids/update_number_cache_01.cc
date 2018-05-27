// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2017 by the deal.II authors
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



// parallel::distributed::Triangulation<dim>::clear forgot to
// update the number cache. check that this is fixed now.

#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include "../tests.h"
#include "coarse_grid_common.h"



template <int dim>
void
test()
{
  parallel::distributed::Triangulation<dim> tr(MPI_COMM_WORLD);

  GridGenerator::hyper_cube(tr);
  tr.refine_global(1);
  deallog << tr.n_global_active_cells() << std::endl;

  tr.clear();
  deallog << tr.n_global_active_cells() << std::endl;
}


int
main(int argc, char* argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  initlog();

  deallog.push("2d");
  test<2>();
  deallog.pop();
}
