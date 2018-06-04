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

  Triangulation<dim>                   triangulation;
  std::unique_ptr<MappingHermite<dim>> mapping_hermite;
  FE_Hermite<dim>                      fe;
  DoFHandler<dim>                      dof_handler;

  std::vector<Vector<double>>                       dof_vectors;
  std::vector<std::vector<boost::optional<double>>> global_dof_values;
  std::vector<std::vector<boost::optional<Tensor<1, dim>>>>
                                                            global_dof_gradients;
  std::vector<std::vector<boost::optional<Tensor<2, dim>>>> global_dof_hessians;
  std::vector<std::vector<boost::optional<Tensor<3, dim>>>>
    global_dof_third_derivatives;
};

template <int dim>
PlotFE<dim>::PlotFE(const unsigned int degree) :
  fe(degree),
  dof_handler(triangulation)
{}

template <int dim>
void
PlotFE<dim>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(2);
  GridTools::distort_random(.1, triangulation);
  mapping_hermite = std_cxx14::make_unique<MappingHermite<dim>>(triangulation);
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

  for (auto &vector : dof_vectors)
    vector.reinit(dof_handler.n_dofs());

  for (unsigned int i = 0; i < dof_vectors.size(); ++i)
    dof_vectors[i](i) = 1.;
}

template <int dim>
void
PlotFE<dim>::check_continuity()
{
  Quadrature<dim> quadrature(dof_handler.get_fe().get_unit_support_points());

  FEValues<dim>      fe_values(*mapping_hermite,
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
  DoFTools::map_dofs_to_support_points(
    *mapping_hermite, dof_handler, support_points);

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
                  AssertThrow(
                    (global_gradient.value() - dof_gradients[local_dof])
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
    {
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          1, DataComponentInterpretation::component_is_scalar);
      std::vector<std::string> dof_names(1,
                                         "dof_" + Utilities::int_to_string(i));
      data_out.add_data_vector(
        dof_handler, dof_vectors[i], dof_names, data_component_interpretation);
    }
  data_out.build_patches(
    mapping_fe_field, 20, DataOut<dim>::curved_inner_cells);

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
  // output_results();
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
  {
    deallog.push("2 4");
    PlotFE<2> plot_fe(4);
    plot_fe.run();
    deallog.pop();
  }
  {
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
  }

  deallog << "OK" << std::endl;

  return 0;
}
