/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/fe_hermite_continuous.h>
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

#include <boost/optional.hpp>

#include <fstream>
#include <iostream>

#include "../tests.h"

using namespace dealii;

template <int dim>
class PlotFE
{
public:
  PlotFE(const unsigned int degree);
  void
  run();

private:
  void
  make_grid();
  void
  setup_system();
  void
  check_continuity();
  void
  output_results() const;

  Triangulation<dim> triangulation;
  // std::unique_ptr<MappingHermite<dim>> mapping_hermite;
  FE_Hermite<dim>           fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints;

  std::vector<Vector<double>>                       dof_vectors;
  std::vector<std::vector<boost::optional<double>>> global_dof_values;
  std::vector<std::vector<boost::optional<Tensor<1, dim>>>>
                                                            global_dof_gradients;
  std::vector<std::vector<boost::optional<Tensor<2, dim>>>> global_dof_hessians;
  std::vector<std::vector<boost::optional<Tensor<3, dim>>>>
    global_dof_third_derivatives;
};

template <int dim>
PlotFE<dim>::PlotFE(const unsigned int degree)
  : fe(degree)
  , dof_handler(triangulation)
{}

template <int dim>
void
PlotFE<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(1);
  triangulation.begin_active()->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();
  /*GridTools::transform(
    [](const Point<dim> &p) -> Point<dim> {
      Point<dim> p_new;
      for (unsigned int i = 0; i < dim; ++i)
        p_new(i) = std::atanh(p(i));
      return p_new;
    },
    triangulation);*/
  GridTools::distort_random(.4, triangulation);
  //  mapping_hermite =
  //  std_cxx14::make_unique<MappingHermite<dim>>(triangulation);
}

template <int dim, int spacedim>
double
evaluate_dof_for_shape_function(
  const FEFaceValuesBase<dim, spacedim> &fe_values,
  const unsigned int                     shape_function,
  const unsigned int                     p,
  const unsigned int                     dof)
{
  if (dim == 2)
    {
      switch (dof % 4)
        {
          case 0:
            return fe_values.shape_value(shape_function, p);
          case 1:
            return fe_values.shape_grad(shape_function, p)[0];
          case 2:
            return fe_values.shape_grad(shape_function, p)[1];
            break;
          case 3:
            return fe_values.shape_hessian(shape_function, p)[0][1];
          default:
            Assert(false, ExcInternalError());
        }
      return fe_values.shape_value(shape_function, p);
    }
  Assert(false, ExcNotImplemented());
  return 0.;
}



template <int dim>
void
make_hanging_node_constraints(const DoFHandler<dim> &    dof_handler,
                              AffineConstraints<double> &constraints)
{
  FE_HermiteContinuous<dim> fe_continuous(dof_handler.get_fe().degree);
  DoFHandler<dim>           dof_handler_continuous;
  dof_handler_continuous.initialize(dof_handler.get_triangulation(),
                                    fe_continuous);
  Assert(fe_continuous.degree == 3, ExcNotImplemented());

  // generate a point on this cell and evaluate the shape functions there
  const Quadrature<dim - 1> quad_face_support(
    fe_continuous.get_unit_face_support_points());

  std::vector<types::global_dof_index> dof_indices_own(
    fe_continuous.dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices_neighbor(
    fe_continuous.dofs_per_cell);

  FESubfaceValues<dim> fe_values_own(fe_continuous,
                                     quad_face_support,
                                     update_quadrature_points | update_values |
                                       update_gradients | update_hessians);
  FEFaceValues<dim>    fe_values_neighbor(fe_continuous,
                                       quad_face_support,
                                       update_quadrature_points |
                                         update_values | update_gradients |
                                         update_hessians);

  // Rule of thumb for FP accuracy, that can be expected for a given
  // polynomial degree. This value is used to cut off values close to
  // zero.
  double eps = 2e-13 * fe_continuous.degree * (dim - 1);

  auto cell_dg = dof_handler.begin_active();
  auto cell_cg = dof_handler_continuous.begin_active();
  for (; cell_dg != dof_handler.end(); ++cell_dg, ++cell_cg)
    {
      cell_dg->get_dof_indices(dof_indices_own);
      std::cout << "Coarse cell: " << cell_dg->center() << std::endl;
      for (unsigned int face_no = 0;
           face_no < GeometryInfo<dim>::faces_per_cell;
           ++face_no)
        {
          if (cell_cg->face(face_no)->has_children())
            {
              for (unsigned int subface_no = 0;
                   subface_no < cell_cg->face(face_no)->n_children();
                   ++subface_no)
                {
                  fe_values_own.reinit(cell_cg, face_no, subface_no);
                  std::cout
                    << "own reinited on: "
                    << cell_cg->face(face_no)->child(subface_no)->center()
                    << std::endl;

                  const auto subface_cell =
                    cell_cg->neighbor_child_on_subface(face_no, subface_no);
                  std::cout << "Fine cell: " << subface_cell->center()
                            << std::endl;
                  fe_values_neighbor.reinit(
                    subface_cell, cell_cg->neighbor_of_neighbor(face_no));
                  std::cout << "neighbor reinited on: "
                            << subface_cell
                                 ->face(cell_cg->neighbor_of_neighbor(face_no))
                                 ->center()
                            << std::endl;
                  for (unsigned int dof = 0; dof < fe_continuous.dofs_per_cell;
                       ++dof)
                    for (unsigned int q = 0; q < fe_continuous.dofs_per_face;
                         ++q)
                      std::cout
                        << "dof values(" << dof << ", " << q
                        << "): " << fe_values_neighbor.shape_value(dof, q)
                        << " " << fe_values_neighbor.shape_grad(dof, q)
                        << std::endl;

                  const auto subface_cell_dg =
                    cell_dg->neighbor_child_on_subface(face_no, subface_no);
                  subface_cell_dg->get_dof_indices(dof_indices_neighbor);
                  for (unsigned int i = 0; i < fe_continuous.dofs_per_face; ++i)
                    {
                      Assert((fe_values_own.quadrature_point(i) -
                              fe_values_neighbor.quadrature_point(i))
                                 .norm() < 1.e-6,
                             ExcInternalError());
                      const auto neighbor_cell_index =
                        fe_continuous.face_to_cell_index(
                          i, cell_cg->neighbor_of_neighbor(face_no));

                      double constrained_value =
                        evaluate_dof_for_shape_function(fe_values_neighbor,
                                                        neighbor_cell_index,
                                                        i,
                                                        i);
                      std::cout
                        << "neighbor cell index: " << neighbor_cell_index
                        << std::endl;
                      std::cout << "constrained value: " << constrained_value
                                << " " << i
                                << fe_values_neighbor.quadrature_point(i)
                                << std::endl;

                      if (!constraints.is_constrained(
                            dof_indices_neighbor[neighbor_cell_index]))
                        {
                          constraints.add_line(
                            dof_indices_neighbor[neighbor_cell_index]);
                          for (unsigned int j = 0;
                               j < fe_continuous.dofs_per_face;
                               ++j)
                            {
                              const auto own_cell_index =
                                fe_continuous.face_to_cell_index(j, face_no);
                              double constraining_value =
                                evaluate_dof_for_shape_function(fe_values_own,
                                                                own_cell_index,
                                                                i,
                                                                i);

                              std::cout
                                << "constraining value: " << constraining_value
                                << " " << j << " "
                                << fe_values_own.quadrature_point(j)
                                << std::endl;

                              if (constrained_value > 0)
                                constraints.add_entry(
                                  dof_indices_neighbor[neighbor_cell_index],
                                  dof_indices_own[own_cell_index],
                                  constraining_value / constrained_value);
                            }
                        }
                    }
                }
            }
        }
    }
}


template <int dim>
void
make_continuity_constraints(const DoFHandler<dim> &    dof_handler,
                            AffineConstraints<double> &constraints)
{
  const FiniteElement<dim> &fe            = dof_handler.get_fe();
  const unsigned int        dofs_per_cell = fe.dofs_per_cell;

  const Quadrature<dim> support(fe.get_unit_support_points());

  FEValues<dim> fe_values_own(fe,
                              support,
                              update_quadrature_points | update_values |
                                update_gradients | update_hessians);
  FEValues<dim> fe_values_neighbor(fe,
                                   support,
                                   update_quadrature_points | update_values |
                                     update_gradients | update_hessians);

  std::vector<types::global_dof_index> dof_indices_own(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices_neighbor(dofs_per_cell);

  std::vector<Point<dim>> points_own;
  std::vector<Point<dim>> points_neighbor;

  std::vector<std::set<typename DoFHandler<dim>::active_cell_iterator>>
    vertex_to_cell_map(dof_handler.get_triangulation().n_vertices());
  {
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
        vertex_to_cell_map[cell->vertex_index(i)].insert(cell);

    // Take care of hanging nodes
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
          if ((cell->at_boundary(i) == false) && (cell->neighbor(i)->active()))
            {
              const auto &adjacent_cell = cell->neighbor(i);
              for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_face;
                   ++j)
                vertex_to_cell_map[cell->face(i)->vertex_index(j)].insert(
                  adjacent_cell);
            }

        // in 3d also loop over the edges
        if (dim == 3)
          {
            for (unsigned int i = 0; i < GeometryInfo<dim>::lines_per_cell; ++i)
              if (cell->line(i)->has_children())
                // the only place where this vertex could have been
                // hiding is on the mid-edge point of the edge we
                // are looking at
                vertex_to_cell_map[cell->line(i)->child(0)->vertex_index(1)]
                  .insert(cell);
          }
      }
  }

  for (const auto &cells : vertex_to_cell_map)
    for (auto cell_it = cells.begin(); cell_it != cells.end(); ++cell_it)
      {
        const auto &cell = *cell_it;
        std::cout << "Cell center: " << cell->center() << std::endl;
        cell->get_dof_indices(dof_indices_own);
        fe_values_own.reinit(cell);
        points_own = fe_values_own.get_quadrature_points();

        const auto own_index = cell->index();

        auto neighbor_cell_it = cell_it;
        ++neighbor_cell_it;
        for (; neighbor_cell_it != cells.end(); ++neighbor_cell_it)
          {
            const auto &neighbor_cell = *neighbor_cell_it;
            std::cout << "Neighbor center: " << neighbor_cell->center()
                      << std::endl;
            const auto neighbor_index = neighbor_cell->index();
            neighbor_cell->get_dof_indices(dof_indices_neighbor);
            {
              fe_values_neighbor.reinit(neighbor_cell);
              points_neighbor = fe_values_neighbor.get_quadrature_points();

              for (unsigned int i = 0; i < dofs_per_cell;
                   i += Utilities::pow(2, dim))
                for (unsigned int j = 0; j < dofs_per_cell;
                     j += Utilities::pow(2, dim))
                  if ((points_own[i] - points_neighbor[j]).norm_square() <
                      1.e-12)
                    {
                      const auto dof_indices_own_start = dof_indices_own[i];
                      const auto dof_indices_neighbor_start =
                        dof_indices_neighbor[j];
                      if (!constraints.is_constrained(
                            dof_indices_neighbor_start))
                        {
                          constraints.add_line(dof_indices_neighbor_start);
                          constraints.add_entry(
                            dof_indices_neighbor_start,
                            dof_indices_own_start,
                            fe_values_own.shape_value(i, i) /
                              fe_values_neighbor.shape_value(j, j));
                        }

                      FullMatrix<double> own_gradient_evaluations(dim, dim);
                      own_gradient_evaluations[0][0] =
                        fe_values_own.shape_grad(i + 1, i + 1)[0];
                      own_gradient_evaluations[1][0] =
                        fe_values_own.shape_grad(i + 1, i + 1)[1];
                      own_gradient_evaluations[0][1] =
                        fe_values_own.shape_grad(i + 2, i + 2)[0];
                      own_gradient_evaluations[1][1] =
                        fe_values_own.shape_grad(i + 2, i + 2)[1];
                      own_gradient_evaluations.print(std::cout);

                      FullMatrix<double> neighbor_gradient_evaluations(dim,
                                                                       dim);
                      neighbor_gradient_evaluations[0][0] =
                        fe_values_neighbor.shape_grad(j + 1, j + 1)[0];
                      neighbor_gradient_evaluations[1][0] =
                        fe_values_neighbor.shape_grad(j + 1, j + 1)[1];
                      neighbor_gradient_evaluations[0][1] =
                        fe_values_neighbor.shape_grad(j + 2, j + 2)[0];
                      neighbor_gradient_evaluations[1][1] =
                        fe_values_neighbor.shape_grad(j + 2, j + 2)[1];
                      neighbor_gradient_evaluations.print(std::cout);

                      FullMatrix<double> inverse_neighbor_gradient_evaluations(
                        dim, dim);
                      inverse_neighbor_gradient_evaluations.invert(
                        neighbor_gradient_evaluations);

                      FullMatrix<double> constraint_matrix(dim, dim);
                      inverse_neighbor_gradient_evaluations.mmult(
                        constraint_matrix, own_gradient_evaluations);

                      for (unsigned int k = 0; k < dim; ++k)
                        if (!constraints.is_constrained(
                              dof_indices_neighbor_start + k + 1))
                          {
                            constraints.add_line(dof_indices_neighbor_start +
                                                 k + 1);
                            for (unsigned int l = 0; l < dim; ++l)
                              constraints.add_entry(dof_indices_neighbor_start +
                                                      k + 1,
                                                    dof_indices_own_start + l +
                                                      1,
                                                    constraint_matrix[k][l]);
                          }

                      if (!constraints.is_constrained(
                            dof_indices_neighbor_start + 3))
                        {
                          constraints.add_line(dof_indices_neighbor_start + 3);
                          constraints.add_entry(
                            dof_indices_neighbor_start + 3,
                            dof_indices_own_start + 3,
                            fe_values_own.shape_hessian(i + 3, i + 3)[0][1] /
                              fe_values_neighbor.shape_hessian(j + 3,
                                                               j + 3)[0][1]);
                        }
                    }
            }
          }
      }
}

template <int dim>
void
PlotFE<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  types::global_dof_index n_dofs = dof_handler.n_dofs();
  dof_vectors.resize(n_dofs);
  global_dof_values.resize(n_dofs,
                           std::vector<boost::optional<double>>(n_dofs));
  global_dof_gradients.resize(
    n_dofs, std::vector<boost::optional<Tensor<1, dim>>>(n_dofs));
  global_dof_hessians.resize(
    n_dofs, std::vector<boost::optional<Tensor<2, dim>>>(n_dofs));
  global_dof_third_derivatives.resize(
    n_dofs, std::vector<boost::optional<Tensor<3, dim>>>(n_dofs));

  constraints.clear();
  std::cout << "continuity constrainnts" << std::endl;
  make_continuity_constraints(dof_handler, constraints);
  std::cout << "hanging node constraints" << std::endl;
  make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();
  constraints.print(std::cout);

  for (auto &vector : dof_vectors)
    vector.reinit(dof_handler.n_dofs());

  for (unsigned int i = 0; i < dof_vectors.size(); ++i)
    {
      dof_vectors[i](i) = 1.;
      constraints.distribute(dof_vectors[i]);
    }
}

template <int dim>
void
PlotFE<dim>::check_continuity()
{
  Quadrature<dim> quadrature(dof_handler.get_fe().get_unit_support_points());

  FEValues<dim>      fe_values(/**mapping_hermite,*/
                          dof_handler.get_fe(),
                          quadrature,
                          update_quadrature_points | update_values |
                            update_gradients /*| update_hessians*/);
  const unsigned int dpc = dof_handler.get_fe().dofs_per_cell;

  std::vector<double>         dof_values(dpc);
  std::vector<Tensor<1, dim>> dof_gradients(dpc);
  std::vector<Tensor<2, dim>> dof_hessians(dpc);
  std::vector<Tensor<3, dim>> dof_third_derivatives(dpc);

  std::vector<types::global_dof_index> ldi(dpc);

  std::map<types::global_dof_index, Point<dim>> support_points;
  MappingQ1<dim>                                mapping;
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(ldi);
      std::cout << "Cell center: " << cell->center() << std::endl;
      for (const auto &global_dof : ldi)
        {
          fe_values.get_function_values(dof_vectors[global_dof], dof_values);
          fe_values.get_function_gradients(dof_vectors[global_dof],
                                           dof_gradients);
          // fe_values.get_function_hessians(dof_vectors[global_dof],
          // dof_hessians);
          // fe_values.get_function_third_derivatives(dof_vectors[global_dof],
          // dof_third_derivatives);

          for (unsigned int local_dof = 0; local_dof < dpc; ++local_dof)
            {
              if (auto &global_value =
                    global_dof_values.at(global_dof).at(ldi[local_dof]))
                {
                  AssertThrow(std::abs(global_value.value() -
                                       dof_values[local_dof]) < 1.e-10,
                              ExcInternalError());
                }
              else
                {
                  global_value = dof_values[local_dof];
                }

              if (auto &global_gradient =
                    global_dof_gradients.at(global_dof).at(ldi[local_dof]))
                {
                  AssertThrow((global_gradient.value() -
                               dof_gradients[local_dof])
                                  .norm() < 1.e-10,
                              ExcInternalError());
                }
              else
                {
                  global_gradient = dof_gradients[local_dof];
                }

              if (auto &global_hessian =
                    global_dof_hessians[global_dof][ldi[local_dof]])
                {
                  for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = i + 1; j < dim; ++j)
                      AssertThrow(std::abs(global_hessian.value()[i][j] -
                                           dof_hessians[local_dof][i][j]) <
                                    1.e-10,
                                  ExcInternalError());
                }
              else
                {
                  global_hessian = dof_hessians[local_dof];
                }

              if (auto &global_third_derivatives =
                    global_dof_third_derivatives[global_dof][ldi[local_dof]])
                {
                  for (unsigned int i = 0; i < dim; ++i)
                    for (unsigned int j = i + 1; j < dim; ++j)
                      for (unsigned int k = j + 1; k < dim; ++k)
                        AssertThrow(
                          std::abs(global_third_derivatives.value()[i][j][k] -
                                   dof_third_derivatives[local_dof][i][j][k]) <
                            1.e-10,
                          ExcInternalError());
                }
              else
                {
                  global_third_derivatives = dof_third_derivatives[local_dof];
                }
            }
        }
    }
}

template <int dim>
void
PlotFE<dim>::output_results() const
{
  MappingHermite<dim> mapping_fe_field(triangulation);
  DataOut<dim>        data_out;

  for (unsigned int i = 0; i < dof_vectors.size(); ++i)
    if (!constraints.is_constrained(i))
      {
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(
            1, DataComponentInterpretation::component_is_scalar);
        std::vector<std::string> dof_names(1,
                                           "dof_" +
                                             Utilities::int_to_string(i, 4));
        data_out.add_data_vector(dof_handler,
                                 dof_vectors[i],
                                 dof_names,
                                 data_component_interpretation);
      }
  data_out.build_patches(20);
  /*  data_out.build_patches(mapping_fe_field,
                           20,
                           DataOut<dim>::curved_inner_cells);*/

  std::ofstream filename("solution.vtk");
  data_out.write_vtk(filename);
}

template <int dim>
void
PlotFE<dim>::run()
{
  make_grid();
  setup_system();
  check_continuity();
  output_results();
}

int
main()
{
  initlog();
  MultithreadInfo::set_thread_limit(1);
  {
    deallog.push("2 3");
    PlotFE<2> plot_fe(3);
    plot_fe.run();
    deallog.pop();
  }
  /*{
    deallog.push("2 4");
    PlotFE<2> plot_fe(4);
    plot_fe.run();
    deallog.pop();
  }*/
  /*{
    deallog.push("3 3");
    PlotFE<3> plot_fe(3);
    plot_fe.run();
    deallog.pop();
  }
  {
    deallog.push("3 4");
    PlotFE<3> plot_fe(4);
    plot_fe.run();
    deallog.pop();
  }*/

  deallog << "OK" << std::endl;

  return 0;
}
