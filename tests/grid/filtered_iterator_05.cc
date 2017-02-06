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

/*
 * Test that copying a FilteredIterator works
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include "../tests.h"

template<int dim>
void
test()
{
  Triangulation<dim> triangulation
  (Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::hyper_L(triangulation, -1, 1);
  triangulation.refine_global(1);
  DoFHandler<dim> dof_handler(triangulation);

  FE_DGQ<2> fe(1);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs(fe);

  FilteredIterator<typename DoFHandler<dim>::level_cell_iterator> begin
  (IteratorFilters::LocallyOwnedLevelCell(), dof_handler.begin());
  FilteredIterator<typename DoFHandler<dim>::level_cell_iterator> end = begin;

  deallog << "OK" << std::endl;
}

int main(int argc, char *argv[])
{
  initlog();

  try
    {
      test<2>();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
