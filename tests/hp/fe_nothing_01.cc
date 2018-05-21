// ---------------------------------------------------------------------
//
// Copyright (C) 2009 - 2017 by the deal.II authors
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

// test that FE_Nothing works as intended: we used to abort in the
// computation of face domination relationships

#include "../tests.h"
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

template <int dim>
void
test()
{
  Triangulation<dim> triangulation;
  GridGenerator ::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(1);

  hp::FECollection<dim> fe_collection;
  fe_collection.push_back(FE_Q<dim>(1));
  fe_collection.push_back(FE_Nothing<dim>());

  hp::DoFHandler<dim> dof_handler(triangulation);

  // loop over cells, and set cells
  // within a circle to be of type
  // FE_Nothing, while outside the
  // circle to be of type FE_Q(1)
  typename hp::DoFHandler<dim>::active_cell_iterator cell
    = dof_handler.begin_active(),
    endc= dof_handler.end();

  for(; cell != endc; cell++)
    {
      Point<dim> center= cell->center();
      if(std::sqrt(center.square()) < 0.5)
        cell->set_active_fe_index(1);
      else
        cell->set_active_fe_index(0);
    }

  // Attempt to distribute dofs.
  // Fails here with assertion from
  // hp_vertex_dof_identities() and
  // after that is fixed in face
  // domination computation.
  dof_handler.distribute_dofs(fe_collection);

  deallog << "   Number of active cells:       "
          << triangulation.n_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;
}

int
main()
{
  std::ofstream logfile("output");
  logfile.precision(2);

  deallog.attach(logfile);

  test<1>();
  test<2>();
  test<3>();

  deallog << "OK" << std::endl;
}
