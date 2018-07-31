/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2018 by the deal.II authors
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
 * Test that FE_Hermite::convert_generalized_support_point_values_to_dof_values
 * works correctly, i.e. linear tensor product polynomials are approximated
 * exatly.
 */

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_hermite.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

template <int dim>
double
linear_shape_function(const Point<dim> &, const unsigned int)
{
  Assert(false, ExcNotImplemented());
  return {};
}

template <>
double
linear_shape_function<2>(const Point<2> &p, const unsigned int n)
{
  switch (n)
    {
      case 0:
        return 1;
      case 1:
        return p[0];
      case 2:
        return p[1];
      case 3:
        return p[0] * p[1];
      default:
        AssertIndexRange(n, 4);
    }
  return {};
}

template <>
double
linear_shape_function<3>(const Point<3> &p, const unsigned int n)
{
  switch (n)
    {
      case 0:
        return 1;
      case 1:
        return p[0];
      case 2:
        return p[1];
      case 3:
        return p[2];
      case 4:
        return p[0] * p[1];
      case 5:
        return p[0] * p[2];
      case 6:
        return p[1] * p[2];
      case 7:
        return p[0] * p[1] * p[2];
      default:
        AssertIndexRange(n, 8);
    }
  return {};
}

template <int dim>
class LinearFunction : public Function<dim>
{
public:
  LinearFunction(const unsigned int n_)
    : n(n_)
  {}

  double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    return linear_shape_function<dim>(p, n);
  }

private:
  const unsigned int n;
};

template <int dim>
void
test()
{
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(FE_Hermite<dim>(3));

  const unsigned int n_functions = Utilities::pow(2, dim);

  const FiniteElement<dim> &fe = dof_handler.get_fe();

  const unsigned int             dofs_per_cell  = fe.dofs_per_cell;
  const std::vector<Point<dim>> &support_points = fe.get_unit_support_points();
  AssertDimension(dofs_per_cell, support_points.size())

    QGauss<dim>
      q_gauss(4);

  for (unsigned int i = 0; i < n_functions; ++i)
    {
      std::vector<Vector<double>> values_to_interpolate(dofs_per_cell,
                                                        Vector<double>(1));

      for (unsigned int dof = 0; dof < dofs_per_cell; ++dof)
        values_to_interpolate[dof] =
          linear_shape_function(support_points[dof], i);

      std::vector<double> interpolated_values(dofs_per_cell);
      fe.convert_generalized_support_point_values_to_dof_values(
        values_to_interpolate, interpolated_values);

      deallog.push(Utilities::int_to_string(i));
      for (unsigned int q = 0; q < q_gauss.size(); ++q)
        {
          double fe_value = 0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            fe_value +=
              fe.shape_value(i, q_gauss.point(q)) * interpolated_values[i];
          deallog << "(" << q << "," << fe_value << ") " << std::endl;
          AssertThrow(std::abs(fe_value -
                               linear_shape_function(q_gauss.point(q), i)) <
                        1.e-10,
                      ExcInternalError());
        }
      deallog.pop();
      deallog << std::endl;
    }
  deallog << "OK" << std::endl;
}

int
main()
{
  initlog();
  deallog.precision(10);

  deallog.push("2");
  test<2>();
  deallog.pop();
  deallog.push("3");
  test<3>();
  deallog.pop();

  return 0;
}
