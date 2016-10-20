// ---------------------------------------------------------------------
//
// Copyright (C) 2006 - 2015 by the deal.II authors
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


// check VectorTools::project for TrilinosWrappers::MPI::Vector arguments


#include "../tests.h"
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/distributed/tria.h>

#include <fstream>


// define the multi-linear function x or x*y or x*y*z that we will
// subsequently project onto the ansatz space
template <int dim>
class F : public Function<dim>
{
public:
  F (unsigned int components=1)
    : Function<dim>(components)
  {}

  virtual double value (const Point<dim> &p,
                        const unsigned int component) const
  {
    double s = 1;
    for (unsigned int i=0; i<dim; ++i)
      s *= p[i];
    return s*(component+1.);
  }
};


template<int dim>
void test()
{
  const MPI_Comm mpi_communicator (MPI_COMM_WORLD);
  parallel::distributed::Triangulation<dim> tria (mpi_communicator);

  GridGenerator::hyper_cube (tria);
  tria.refine_global (2);

  FESystem<dim> fe (FE_Q<dim>(1), dim);
  DoFHandler<dim> dh (tria);
  dh.distribute_dofs (fe);

  const IndexSet locally_owned_dofs = dh.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dh, locally_relevant_dofs);

  TrilinosWrappers::MPI::Vector v_distributed(locally_owned_dofs, mpi_communicator);
  TrilinosWrappers::MPI::Vector v_ghosted(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  ConstraintMatrix cm;
  cm.reinit(locally_relevant_dofs);
  cm.close ();
  VectorTools::project_distributed<dim, TrilinosWrappers::MPI::Vector, dim, dim, 1>
  (dh, cm, QGauss<dim>(3), F<dim>(dim), v_distributed);

  v_ghosted = v_distributed;

  for (typename DoFHandler<dim>::active_cell_iterator cell=dh.begin_active();
       cell != dh.end(); ++cell)
    if (cell->is_locally_owned())
      for (unsigned int i=0; i<GeometryInfo<dim>::vertices_per_cell; ++i)
        {
          // check that the error is
          // somewhat small. it won't
          // be zero since we project
          // and do not interpolate
          Assert (std::fabs (v_ghosted(cell->vertex_dof_index(i,0)) -
                             F<dim>(dim).value (cell->vertex(i),0))
                  < 1e-4,
                  ExcInternalError());
          deallog << cell->vertex(i) << ' ' << v_ghosted(cell->vertex_dof_index(i,0))
                  << std::endl;
        }
}


int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init_finalize(argc, argv, 1);
  mpi_initlog();

  try
    {
      test<2>();
      test<3>();
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
