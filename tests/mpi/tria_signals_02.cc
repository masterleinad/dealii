// ---------------------------------------------------------------------
//
// Copyright (C) 2015 - 2017 by the deal.II authors
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

#include "../tests.h"
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <functional>
#include <ostream>

// Test on whether signals post_refinement_on_cell and pre_coarsening_on_cell
// could catch all cell changes.
// The test is designed to count cell number increase and decrease in signal
// calls and then compare the result against n_active_cells reported by Tria
// object. Absolute value change in n_active_cells is not concerned in this test.

template <int dim, int spacedim>
class SignalListener
{
public:
  SignalListener(Triangulation<dim, spacedim> & tria_in)
    : n_active_cells(tria_in.n_active_cells()), tria(tria_in)
  {
    tria_in.signals.post_refinement_on_cell.connect(
      std::bind(&SignalListener<dim, spacedim>::count_on_refine,
                this,
                std::placeholders::_1));

    tria_in.signals.pre_coarsening_on_cell.connect(
      std::bind(&SignalListener<dim, spacedim>::count_on_coarsen,
                this,
                std::placeholders::_1));
  }

  int
  n_active_cell_gap()
  {
    return (n_active_cells - static_cast<int>(tria.n_active_cells()));
  }

private:
  void
  count_on_refine(
    const typename Triangulation<dim, spacedim>::cell_iterator & cell)
  {
    n_active_cells += cell->n_children();
    --n_active_cells;

    return;
  }

  void
  count_on_coarsen(
    const typename Triangulation<dim, spacedim>::cell_iterator & cell)
  {
    ++n_active_cells;
    n_active_cells -= cell->n_children();

    return;
  }

  int                                  n_active_cells;
  const Triangulation<dim, spacedim> & tria;
};

template <int dim, int spacedim>
void
test()
{
  typedef parallel::distributed::Triangulation<dim, spacedim> TriaType;

  {
    const std::string prefix = Utilities::int_to_string(dim, 1) + "d-"
                               + Utilities::int_to_string(spacedim, 1) + "d";
    deallog.push(prefix.c_str());
  }

  TriaType tria(MPI_COMM_WORLD);

  GridGenerator::hyper_cube(tria);
  SignalListener<dim, spacedim> count_cell_via_signal(tria);

  tria.refine_global(2);

  deallog << "n_cell_gap after refine : "
          << count_cell_via_signal.n_active_cell_gap() << std::endl;

  // Test signal on coarsening
  {
    typename TriaType::active_cell_iterator       cell = tria.begin_active();
    const typename TriaType::active_cell_iterator endc = tria.end();

    for(; cell != endc; ++cell)
      {
        cell->set_coarsen_flag();
      }
    tria.execute_coarsening_and_refinement();
  }

  deallog << "n_cell_gap after coarsen : "
          << count_cell_via_signal.n_active_cell_gap() << std::endl;

  deallog.pop();
  return;
}

int
main(int argc, char * argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, /* int max_num_threads */ 1);
  MPILogInitAll log;

  // parallel::distributed::Triangulation<1, spacedim> is not valid.
  {
    const int dim      = 2;
    const int spacedim = 2;
    test<dim, spacedim>();
  }

  {
    const int dim      = 2;
    const int spacedim = 3;
    test<dim, spacedim>();
  }

  {
    const int dim      = 3;
    const int spacedim = 3;
    test<dim, spacedim>();
  }

  return (0);
}
