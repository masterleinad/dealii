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

// tests matrix-free face evaluation, matrix-vector products as compared to
// the same implementation with MeshWorker. This example uses a hyperball mesh
// with both hanging nodes and curved boundaries

#include "../tests.h"
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_dgq.h>

std::ofstream logfile("output");

#include "matrix_vector_faces_common.h"

template <int dim, int fe_degree>
void
test()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_ball(tria);
  tria.refine_global(4 - dim);
  tria.begin_active()->set_refine_flag();
  tria.last()->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  FE_DGQ<dim>     fe(fe_degree);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  ConstraintMatrix constraints;
  constraints.close();

  // test with threads enabled as well
  do_test<dim, fe_degree, fe_degree + 1, double>(dof, constraints, true);
}
