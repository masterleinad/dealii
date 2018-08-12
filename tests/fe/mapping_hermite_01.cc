// ---------------------------------------------------------------------
//
// Copyright (C) 2018 by the deal.II authors
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

// Test that Mapping_Hermite is actually C1 (checked visually).

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_hermite.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

#include "../tests.h"

// Test

using namespace dealii;

template <int dim>
void
test()
{
  Triangulation<dim> triangulation;
  Point<dim>         p0;
  Point<dim>         p1;
  p1(0) = 2.;
  p1(1) = 1.;
  std::vector<unsigned int> subdivisions(2, 1);
  subdivisions[0] = 2;
  /*GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            subdivisions,
                                            p0,
                                            p1);*/
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(1);
  triangulation.begin_active()->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();
  GridTools::distort_random(.4, triangulation);

  MappingHermite<dim> mapping_fe_field(triangulation);
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(mapping_fe_field.get_dof_handler());
    Vector<float> dummy(triangulation.n_active_cells());
    data_out.add_data_vector(dummy, "dummy");
    data_out.build_patches(mapping_fe_field,
                           20,
                           DataOut<dim>::curved_inner_cells);

    std::ofstream filename("solution-mapped" + Utilities::int_to_string(dim) +
                           ".vtk");
    data_out.write_vtk(filename);
    data_out.write_vtk(deallog.get_file_stream());
  }
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(mapping_fe_field.get_dof_handler());
    Vector<float> dummy(triangulation.n_active_cells());
    data_out.add_data_vector(dummy, "dummy");
    data_out.build_patches(20);

    std::ofstream filename("solution" + Utilities::int_to_string(dim) + ".vtk");

    data_out.write_vtk(filename);
  }
}

int
main()
{
  initlog();
  {
    test<2>();
    // test<3>();
  }

  return 0;
}
