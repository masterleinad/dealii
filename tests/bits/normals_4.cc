// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2018 by the deal.II authors
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

// integrating the normals over the surface of any cell (distorted or
// not) should yield a zero vector. (To prove this, use the divergence
// theorem.) check that this is indeed so for the hypercube in 2d and 3d
//
// in contrast to normals_1, do this with a cartesian mapping
// associated with the geometry

#include "../tests.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

template <int dim>
void
check(const Triangulation<dim> &tria)
{
  MappingCartesian<dim> mapping;
  FE_Q<dim>             fe(1);
  DoFHandler<dim>       dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  QGauss<dim - 1> q_face(3);

  FEFaceValues<dim> fe_face_values(
    mapping, fe, q_face, update_normal_vectors | update_JxW_values);
  FESubfaceValues<dim> fe_subface_values(
    mapping, fe, q_face, update_normal_vectors | update_JxW_values);

  for(typename DoFHandler<dim>::active_cell_iterator cell
      = dof_handler.begin_active();
      cell != dof_handler.end();
      ++cell)
    {
      Tensor<1, dim> n1, n2;

      // first integrate over faces
      // and make sure that the
      // result of the integration is
      // close to zero
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
          fe_face_values.reinit(cell, f);
          for(unsigned int q = 0; q < q_face.size(); ++q)
            n1 += fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
        }
      Assert(n1 * n1 < 1e-24, ExcInternalError());
      deallog << cell << " face integration is ok: " << std::sqrt(n1 * n1)
              << std::endl;

      // now same for subface
      // integration
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        for(unsigned int sf = 0; sf < GeometryInfo<dim>::max_children_per_face;
            ++sf)
          {
            fe_subface_values.reinit(cell, f, sf);
            for(unsigned int q = 0; q < q_face.size(); ++q)
              n2 += fe_subface_values.normal_vector(q)
                    * fe_subface_values.JxW(q);
          }
      Assert(n2 * n2 < 1e-24, ExcInternalError());
      deallog << cell << " subface integration is ok: " << std::sqrt(n2 * n2)
              << std::endl;
    }
}

int
main()
{
  initlog();

  {
    Triangulation<2> coarse_grid;
    GridGenerator::hyper_cube(coarse_grid);
    check(coarse_grid);
  }

  {
    Triangulation<3> coarse_grid;
    GridGenerator::hyper_cube(coarse_grid);
    check(coarse_grid);
  }
}
