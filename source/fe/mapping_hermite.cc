// ---------------------------------------------------------------------
//
// Copyright (C) 2001 - 2018 by the deal.II authors
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

#include <deal.II/base/array_view.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/std_cxx14/memory.h>
#include <deal.II/base/tensor_product_polynomials.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_hermite.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/transformations.h>

#include <fstream>
#include <memory>
#include <numeric>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim>
MappingHermiteHelper<dim, spacedim>::MappingHermiteHelper(
  const Triangulation<dim, spacedim> &triangulation,
  const bool                          orthogonalize)
  : dof_handler(std::make_shared<DoFHandler<dim, spacedim>>(triangulation))
{
  reinit(orthogonalize);
}

namespace
{
  template <int dim, int spacedim>
  std::vector<Point<spacedim>>
  compute_intermediate_points(const std::vector<Point<spacedim>> &vertices,
                              const std::vector<Point<dim>> &evaluation_points)
  {
    AssertDimension(vertices.size(), GeometryInfo<dim>::vertices_per_cell);
    const unsigned int           n_points = evaluation_points.size();
    std::vector<Point<spacedim>> intermediate_points(n_points);

    for (unsigned int q = 0; q < n_points; ++q)
      for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
        intermediate_points[q] +=
          vertices[i] *
          GeometryInfo<dim>::d_linear_shape_function(evaluation_points[q], i);
    return intermediate_points;
  }
} // namespace

template <int dim, int spacedim>
void
MappingHermiteHelper<dim, spacedim>::reinit(const bool orthogonalize)
{
  if (dof_handler->has_active_dofs())
    dof_handler->distribute_dofs(dof_handler->get_fe());
  else
    dof_handler->distribute_dofs(
      FESystem<dim, spacedim>(FE_Hermite<dim, spacedim>(4), dim));
  IndexSet active_dofs;
  DoFTools::extract_locally_active_dofs(*dof_handler, active_dofs);
  hermite_vector.reinit(active_dofs, MPI_COMM_WORLD);
  {
    const FiniteElement<dim, spacedim> &       fe = dof_handler->get_fe();
    const unsigned int                         dofs_per_cell = fe.dofs_per_cell;
    LinearAlgebra::distributed::Vector<double> n_values(hermite_vector);
    hermite_vector = 0.;
    n_values       = 0.;

    const unsigned int component_dofs_per_vertex = Utilities::pow(2, dim);
    std::cout << "component dofs per vertex: " << component_dofs_per_vertex
              << std::endl;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Point<dim>> unit_support_points = fe.get_unit_support_points();
    AssertDimension(unit_support_points.size(),
                    dim * Utilities::pow(fe.degree + 1, dim));

    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);

          // FESystem doesn't like multiple support points in one place so don't
          // use it for interpolating
          std::vector<std::vector<Vector<double>>> values_to_interpolate(
            dim,
            std::vector<Vector<double>>(dofs_per_cell / dim,
                                        Vector<double>(1)));
          std::vector<Point<spacedim>> cell_vertices(
            GeometryInfo<dim>::vertices_per_cell);
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            cell_vertices[v] = cell->vertex(v);

          std::vector<Point<spacedim>> intermediate_points =
            compute_intermediate_points(cell_vertices, unit_support_points);
          /*for (unsigned int component = 0; component < dim; ++component)
            for (unsigned int vertex = 0;
                 vertex < GeometryInfo<dim>::vertices_per_cell;
                 ++vertex)
              for (unsigned int vertex_dof = 0;
                   vertex_dof < component_dofs_per_vertex;
                   ++vertex_dof)
                vertex_values[component * dofs_per_cell / dim +
                              vertex * component_dofs_per_vertex + vertex_dof] =
                  cell->vertex(vertex);*/

          for (unsigned int i = 0; i < intermediate_points.size(); ++i)
            {
              const unsigned int component =
                fe.system_to_component_index(i).first;
              const unsigned int within_base =
                fe.system_to_component_index(i).second;
              values_to_interpolate[component][within_base](0) =
                intermediate_points[i](component);
              std::cout << "(" << component << ", " << within_base
                        << "): " << intermediate_points[i] << std::endl;
            }

          std::vector<std::vector<double>> interpolated_values(
            dim, std::vector<double>(dofs_per_cell / dim));
          for (unsigned int d = 0; d < dim; ++d)
            fe.base_element(0)
              .convert_generalized_support_point_values_to_dof_values(
                values_to_interpolate[d], interpolated_values[d]);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component =
                fe.system_to_component_index(i).first;
              const unsigned int within_base =
                fe.system_to_component_index(i).second;

              hermite_vector(local_dof_indices[i]) +=
                interpolated_values[component][within_base];
              ++n_values(local_dof_indices[i]);
              std::cout << i << ": " << hermite_vector(local_dof_indices[i])
                        << std::endl;
            }
        }

    for (unsigned int i = 0; i < dof_handler->n_dofs(); ++i)
      {
        hermite_vector(i) /= n_values(i);
      }

    if (orthogonalize)
      orthogonalize_gradients();

    AffineConstraints<double> constraints;
    constraints.clear();
    /*FE_Hermite<dim, spacedim>::make_continuity_constraints(*dof_handler,
                                                           constraints);
    std::ofstream print_continuity("continuity");
    constraints.print(print_continuity);
    std::cout << "Hanging node constraints" << std::endl;
    FE_Hermite<dim, spacedim>::make_hanging_node_constraints(*dof_handler,
                                                             constraints);
    std::ofstream print_combined("combined");
    constraints.print(print_combined);*/
    constraints.close();

    constraints.print(std::cout);

    constraints.distribute(hermite_vector);
  }
}



template <int dim, int spacedim>
void
MappingHermiteHelper<dim, spacedim>::orthogonalize_gradients()
{
  const unsigned int dofs_per_component =
    dof_handler->get_fe().dofs_per_cell / dim;
  const unsigned int dofs_per_vertex = 1 << dim;

  std::vector<types::global_dof_index> ldi(dofs_per_component * dim);

  for (const auto &cell : dof_handler->active_cell_iterators())
    {
      cell->get_dof_indices(ldi);

      // get angle with respect to x-axis
      for (unsigned int i = 0; i < dofs_per_component; i += dofs_per_vertex)
        {
          if (dim == 2)
            {
              Tensor<1, 3> gradient_1;
              gradient_1[0] = hermite_vector(ldi[i + (1 << 0)]);
              gradient_1[1] =
                hermite_vector(ldi[i + dofs_per_component + (1 << 0)]);

              Tensor<1, 3> gradient_2;
              gradient_2[0] = hermite_vector(ldi[i + (1 << 1)]);
              gradient_2[1] =
                hermite_vector(ldi[i + dofs_per_component + (1 << 1)]);

              const double dot   = gradient_1 * gradient_2;
              const double det   = cross_product_3d(gradient_1, gradient_2)[2];
              const double angle = std::atan2(det, dot);
              const double correction_angle = .5 * angle - .25 * numbers::PI;

              Tensor<1, dim> new_gradient_1;
              new_gradient_1[0] = std::cos(correction_angle) * gradient_1[0] -
                                  std::sin(correction_angle) * gradient_1[1];
              new_gradient_1[1] = std::sin(correction_angle) * gradient_1[0] +
                                  std::cos(correction_angle) * gradient_1[1];

              Tensor<1, dim> new_gradient_2;
              new_gradient_2[0] = std::cos(-correction_angle) * gradient_2[0] -
                                  std::sin(-correction_angle) * gradient_2[1];
              new_gradient_2[1] =
                hermite_vector(ldi[i + dofs_per_component + (1 << 1)]) =
                  std::sin(-correction_angle) * gradient_2[0] +
                  std::cos(-correction_angle) * gradient_2[1];

              Assert(std::abs(new_gradient_1 * new_gradient_2) < 1.e-6,
                     ExcInternalError());
              hermite_vector(ldi[i + (1 << 0)]) = new_gradient_1[0];
              hermite_vector(ldi[i + dofs_per_component + (1 << 0)]) =
                new_gradient_1[1];
              hermite_vector(ldi[i + (1 << 1)]) = new_gradient_2[0];
              hermite_vector(ldi[i + dofs_per_component + (1 << 1)]) =
                new_gradient_2[1];
            }
          else // dim==3
            {
              Tensor<1, 3> gradient_1;
              gradient_1[0] = hermite_vector(ldi[i + (1 << 0)]);
              gradient_1[1] =
                hermite_vector(ldi[i + dofs_per_component + (1 << 0)]);
              gradient_1[1] =
                hermite_vector(ldi[i + 2 * dofs_per_component + (1 << 0)]);

              Tensor<1, 3> gradient_2;
              gradient_2[0] = hermite_vector(ldi[i + (1 << 1)]);
              gradient_2[1] =
                hermite_vector(ldi[i + dofs_per_component + (1 << 1)]);
              gradient_2[2] =
                hermite_vector(ldi[i + 2 * dofs_per_component + (1 << 1)]);

              Tensor<1, 3> gradient_3;
              gradient_3[0] = hermite_vector(ldi[i + (1 << 2)]);
              gradient_3[1] =
                hermite_vector(ldi[i + dofs_per_component + (1 << 2)]);
              gradient_3[2] =
                hermite_vector(ldi[i + 2 * dofs_per_component + (1 << 2)]);

              const double dot = gradient_1 * gradient_2;
              auto normal = Point<3>(cross_product_3d(gradient_1, gradient_2));
              normal /= normal.norm();
              const double angle =
                std::acos(dot / (gradient_1.norm() * gradient_2.norm()));
              const double correction_angle = .5 * angle - .25 * numbers::PI;

              const auto rotation_1 =
                Physics::Transformations::Rotations::rotation_matrix_3d(
                  normal, correction_angle);
              const auto rotation_2 =
                Physics::Transformations::Rotations::rotation_matrix_3d(
                  normal, -correction_angle);

              Tensor<1, 3> new_gradient_1 = rotation_1 * gradient_1;
              Tensor<1, 3> new_gradient_2 = rotation_2 * gradient_2;
              Tensor<1, 3> new_gradient_3 = (normal * gradient_3) * normal;

              Assert(std::abs(new_gradient_1 * new_gradient_2) < 1.e-6,
                     ExcInternalError());
              Assert(std::abs(new_gradient_1 * new_gradient_3) < 1.e-6,
                     ExcInternalError());
              Assert(std::abs(new_gradient_2 * new_gradient_3) < 1.e-6,
                     ExcInternalError());

              hermite_vector(ldi[i + (1 << 0)]) = new_gradient_1[0];
              hermite_vector(ldi[i + dofs_per_component + (1 << 0)]) =
                new_gradient_1[1];
              hermite_vector(ldi[i + 2 * dofs_per_component + (1 << 0)]) =
                new_gradient_1[2];
              hermite_vector(ldi[i + (1 << 1)]) = new_gradient_2[0];
              hermite_vector(ldi[i + dofs_per_component + (1 << 1)]) =
                new_gradient_2[1];
              hermite_vector(ldi[i + 2 * dofs_per_component + (1 << 1)]) =
                new_gradient_2[2];
              hermite_vector(ldi[i + (1 << 2)]) = new_gradient_3[0];
              hermite_vector(ldi[i + dofs_per_component + (1 << 2)]) =
                new_gradient_3[1];
              hermite_vector(ldi[i + 2 * dofs_per_component + (1 << 2)]) =
                new_gradient_3[2];
            }
        }
    }
}



template <int dim, int spacedim>
const DoFHandler<dim, spacedim> &
MappingHermiteHelper<dim, spacedim>::get_dof_handler() const
{
  return *dof_handler;
}

template <int dim, int spacedim>
const LinearAlgebra::distributed::Vector<double> &
MappingHermiteHelper<dim, spacedim>::get_hermite_vector() const
{
  return hermite_vector;
}

template <int dim, int spacedim>
MappingHermite<dim, spacedim>::MappingHermite(
  const Triangulation<dim, spacedim> &triangulation,
  const bool                          orthogonalize)
  : MappingHermiteHelper<dim, spacedim>(triangulation, orthogonalize)
  , MappingFEField<dim,
                   spacedim,
                   LinearAlgebra::distributed::Vector<double>,
                   dealii::DoFHandler<dim, spacedim>>(
      MappingHermiteHelper<dim, spacedim>::get_dof_handler(),
      MappingHermiteHelper<dim, spacedim>::get_hermite_vector())
{}

template <int dim, int spacedim>
void
MappingHermite<dim, spacedim>::reinit(const bool orthogonalize)
{
  MappingHermiteHelper<dim, spacedim>::reinit(orthogonalize);
}

template <int dim, int spacedim>
bool
MappingHermite<dim, spacedim>::preserves_vertex_locations() const
{
  return true;
}

template <int dim, int spacedim>
std::unique_ptr<Mapping<dim, spacedim>>
MappingHermite<dim, spacedim>::clone() const
{
  return std_cxx14::make_unique<MappingHermite<dim, spacedim>>(*this);
}

// explicit instantiations
#include "mapping_hermite.inst"

DEAL_II_NAMESPACE_CLOSE
