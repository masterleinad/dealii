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

// check that processors correctly communicate active_fe_index
// information from their own locally owned cells to the ghosts on
// other processors
// This is the same as communicate_active_fe_indices_01 but using
// METIS as partitioner.

#include "../tests.h"
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/hp/dof_handler.h>

#include <numeric>

template <int dim>
void
test()
{
  parallel::shared::Triangulation<dim> triangulation(
    MPI_COMM_WORLD,
    ::Triangulation<dim>::none,
    false,
    parallel::shared::Triangulation<dim>::partition_metis);

  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(3);

  hp::FECollection<dim> fe;
  for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
       ++i)
    fe.push_back(FE_Q<dim>(1));

  hp::DoFHandler<dim> dof_handler(triangulation);

  // set the active_fe_index on all locally active cells equal to the
  // subdomain_id. we can later verify this equality also on ghost
  // cells
  for (auto cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(cell->subdomain_id());

  dof_handler.distribute_dofs(fe);

  deallog << "n_dofs: " << dof_handler.n_dofs() << std::endl;
  deallog << "n_locally_owned_dofs: " << dof_handler.n_locally_owned_dofs()
          << std::endl;

  // verify that the information has been communicated between processors
  for (auto cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        Assert(cell->active_fe_index() == cell->subdomain_id(),
               ExcInternalError());
      if (cell->is_ghost())
        Assert(cell->active_fe_index() == cell->subdomain_id(),
               ExcInternalError());

      // in the current mode for p::s::Tria, every cell is either
      // locally owned or a ghost
      Assert(cell->is_locally_owned() || cell->is_ghost(), ExcInternalError());
    }
}

int
main(int argc, char* argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPILogInitAll all;

  deallog.push("1d");
  test<1>();
  deallog.pop();

  deallog.push("2d");
  test<2>();
  deallog.pop();

  deallog.push("3d");
  test<3>();
  deallog.pop();
}
