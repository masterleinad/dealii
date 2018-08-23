// ---------------------------------------------------------------------
//
// Copyright (C) 2000 - 2017 by the deal.II authors
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

#include <deal.II/base/polynomials_hermite.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/std_cxx14/memory.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/fe_hermite_continuous.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>

#include <sstream>
#include <vector>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim>
FE_Hermite<dim, spacedim>::FE_Hermite(const unsigned int degree)
  : FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>(
      this->get_polynomials(degree),
      FiniteElementData<dim>(this->get_dpo_vector(degree),
                             1,
                             degree,
                             FiniteElementData<dim>::L2),
      std::vector<bool>(1, false),
      std::vector<ComponentMask>(1, std::vector<bool>(1, true)))
{
  std::vector<Point<1>> points_1d;
  points_1d.reserve(degree - 1);
  for (unsigned int i = 0; i < degree - 1; ++i)
    points_1d.push_back(Point<1>(i * 1. / (degree - 2)));
  initialize_unit_support_points(points_1d);
  initialize_unit_face_support_points(points_1d);
}

template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::initialize_unit_support_points(
  const std::vector<Point<1>> &points)
{
  AssertDimension(points.size() + 1, this->degree);

  std::vector<Point<1>> actual_support_points;
  actual_support_points.reserve(this->degree + 1);
  actual_support_points.push_back(points.front());
  actual_support_points.push_back(points.front());
  actual_support_points.push_back(points.back());
  actual_support_points.push_back(points.back());
  actual_support_points.insert(actual_support_points.end(),
                               ++points.begin(),
                               --points.end());

  // We can compute the support points by computing the tensor
  // product of the 1d set of points. We could do this by hand, but it's
  // easier to just re-use functionality that's already been implemented
  // for quadrature formulas.
  const Quadrature<1>   support_1d(actual_support_points);
  const Quadrature<dim> support_quadrature(support_1d); // NOLINT

  const std::vector<unsigned int> &index_map_inverse =
    this->poly_space.get_numbering_inverse();

  // The only thing we have to do is reorder the points from tensor
  // product order to the order in which we enumerate DoFs on cells
  this->unit_support_points.resize(support_quadrature.size());
  for (unsigned int k = 0; k < support_quadrature.size(); ++k)
    this->unit_support_points[index_map_inverse[k]] =
      support_quadrature.point(k);
}

template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::initialize_unit_face_support_points(
  const std::vector<Point<1>> &points)
{
  AssertDimension(points.size() + 1, this->degree);

  const unsigned int degree = this->degree;

  std::vector<Point<1>> actual_support_points;
  actual_support_points.reserve(this->degree + 1);
  actual_support_points.push_back(points.front());
  actual_support_points.push_back(points.front());
  actual_support_points.push_back(points.back());
  actual_support_points.push_back(points.back());
  actual_support_points.insert(actual_support_points.end(),
                               ++points.begin(),
                               --points.end());

  // find renumbering of faces and assign from values of quadrature
  std::vector<unsigned int> dpo(dim, 0);
  for (unsigned int i = 0; i < dpo.size(); ++i)
    dpo[i] = Utilities::pow(2, dim - 1 - i) * Utilities::pow(degree - 3, i);
  const std::vector<unsigned int> face_index_map =
    FETools::general_lexicographic_to_hierarchic<dim - 1>(dpo);

  // We can compute the support points by computing the tensor
  // product of the 1d set of points. We could do this by hand, but it's
  // easier to just re-use functionality that's already been implemented
  // for quadrature formulas.
  const Quadrature<1>       support_1d(actual_support_points);
  const Quadrature<dim - 1> support_quadrature(support_1d); // NOLINT

  // The only thing we have to do is reorder the points from tensor
  // product order to the order in which we enumerate DoFs on cells
  this->unit_face_support_points.resize(support_quadrature.size() * 2);
  for (unsigned int k = 0; k < support_quadrature.size() * 2; ++k)
    {
      this->unit_face_support_points[k] =
        support_quadrature.point(face_index_map[k / 2]);
    }
}

template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::get_interpolation_matrix(
  const FiniteElement<dim, spacedim> &,
  FullMatrix<double> &) const
{
  // no interpolation possible. throw exception, as documentation says
  AssertThrow(
    false,
    (typename FiniteElement<dim, spacedim>::ExcInterpolationNotImplemented()));
}

template <int dim, int spacedim>
const FullMatrix<double> &
FE_Hermite<dim, spacedim>::get_restriction_matrix(
  const unsigned int,
  const RefinementCase<dim> &) const
{
  AssertThrow(false,
              (typename FiniteElement<dim, spacedim>::ExcProjectionVoid()));
  // return dummy, nothing will happen because the base class FE_Q_Base
  // implements lazy evaluation of those matrices
  return this->restriction[0][0];
}

template <int dim, int spacedim>
const FullMatrix<double> &
FE_Hermite<dim, spacedim>::get_prolongation_matrix(
  const unsigned int,
  const RefinementCase<dim> &) const
{
  AssertThrow(false,
              (typename FiniteElement<dim, spacedim>::ExcEmbeddingVoid()));
  // return dummy, nothing will happen because the base class FE_Q_Base
  // implements lazy evaluation of those matrices
  return this->prolongation[0][0];
}

template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::get_face_interpolation_matrix(
  const FiniteElement<dim, spacedim> &source_fe,
  FullMatrix<double> &                interpolation_matrix) const
{
  Assert(dim > 1, ExcImpossibleInDim(1));
  get_subface_interpolation_matrix(source_fe,
                                   numbers::invalid_unsigned_int,
                                   interpolation_matrix);
}



template <int dim, int spacedim>
double
FE_Hermite<dim, spacedim>::evaluate_dof_for_shape_function(
  const FiniteElement<dim, spacedim> &fe,
  const unsigned int                  dof,
  const Point<dim> &                  p,
  const unsigned int                  shape_function)
{
  (void)dof;
  double scale = 1.;
  if (dim == 2 && fe.degree == 3)
    {
      switch (dof % 4)
        {
          case 0:
            return fe.shape_value(shape_function, p);
          case 1:
            return fe.shape_grad(shape_function, p)[0] * scale;
          case 2:
            return fe.shape_grad(shape_function, p)[1] * scale;
            break;
          case 3:
            return fe.shape_grad_grad(shape_function, p)[0][1] * scale * scale;
          default:
            Assert(false, ExcInternalError());
        }
      return fe.shape_value(shape_function, p);
    }
  Assert(false, ExcNotImplemented());
  return 0.;
}



template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::get_subface_interpolation_matrix(
  const FiniteElement<dim, spacedim> &x_source_fe,
  const unsigned int                  subface,
  FullMatrix<double> &                interpolation_matrix) const
{
  (void)x_source_fe;
  (void)subface;
  (void)interpolation_matrix;
  Assert(interpolation_matrix.m() == x_source_fe.dofs_per_face,
         ExcDimensionMismatch(interpolation_matrix.m(),
                              x_source_fe.dofs_per_face));
  // see if source is a FE_Hermite element
  if (const auto source_fe = dynamic_cast<const FE_Hermite<2> *>(&x_source_fe))
    {
      Assert(source_fe->degree == 3, ExcNotImplemented());
      Assert(interpolation_matrix.n() == this->dofs_per_face,
             ExcDimensionMismatch(interpolation_matrix.n(),
                                  this->dofs_per_face));
      Assert(this->degree == 3, ExcNotImplemented());

      // generate a point on this cell and evaluate the shape functions there
      const Quadrature<dim - 1> quad_face_support(
        source_fe->get_unit_face_support_points());

      // Rule of thumb for FP accuracy, that can be expected for a given
      // polynomial degree. This value is used to cut off values close to
      // zero.
      double eps = 2e-13 * this->degree * (dim - 1);

      // compute the the first entries of interpolation matrix by simply taking
      // the value at the support points.
      // TODO: Verify that all faces are the same with respect to
      // these support points. Furthermore, check if something has to
      // be done for the face orientation flag in 3D.
      const Quadrature<dim> subface_quadrature =
        subface == numbers::invalid_unsigned_int ?
          QProjector<dim>::project_to_face(quad_face_support, 0) :
          QProjector<dim>::project_to_subface(quad_face_support, 0, subface);
      for (unsigned int i = 0; i < source_fe->dofs_per_face; ++i)
        {
          // const Point<dim> &p = subface_quadrature.point(i);

          for (unsigned int j = 0; j < this->dofs_per_face; ++j)
            {
              double matrix_entry = 0;
              /*                evaluate_dof_for_shape_function(this->face_to_cell_index(i,
                 0), p, this->face_to_cell_index(j, 0));*/

              // Correct the interpolated value. I.e. if it is close to 1 or
              // 0, make it exactly 1 or 0. Unfortunately, this is required to
              // avoid problems with higher order elements.
              if (std::fabs(matrix_entry - 1.0) < eps)
                matrix_entry = 1.0;
              if (std::fabs(matrix_entry) < eps)
                matrix_entry = 0.0;
              interpolation_matrix(i, j) = matrix_entry;
            }
        }

      // make sure that the row sum of each of the matrices is 1 at this
      // point. this must be so since the shape functions sum up to 1
      /*for (unsigned int j = 0; j < source_fe->dofs_per_face; ++j)
        {
          double sum = 0.;

          for (unsigned int i = 0; i < this->dofs_per_face; ++i)
            sum += interpolation_matrix(j, i);

          Assert(std::fabs(sum - 1) < eps, ExcInternalError());
        }*/
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&x_source_fe) != nullptr)
    {
      // nothing to do here, the FE_Nothing has no degrees of freedom anyway
    }
  else
    AssertThrow(
      false,
      (typename FiniteElement<dim,
                              spacedim>::ExcInterpolationNotImplemented()));
}

template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::
  convert_generalized_support_point_values_to_dof_values(
    const std::vector<Vector<double>> &support_point_values,
    std::vector<double> &              nodal_values) const
{
  AssertDimension(support_point_values.size(), nodal_values.size());
  AssertDimension(nodal_values.size(), this->dofs_per_cell);

  const unsigned int dofs_per_vertex = Utilities::pow(2, dim);
  const unsigned int dofs_per_line =
    Utilities::pow(this->degree - 3, 1) * Utilities::pow(2, dim - 1);
  const unsigned int dofs_per_quad =
    Utilities::pow(this->degree - 3, 2) * Utilities::pow(2, dim - 2);

  std::cout << "input begin" << std::endl;
  for (unsigned int i = 0; i < this->dofs_per_cell; ++i)
    std::cout << i << ": " << support_point_values[i](0) << std::endl;
  std::cout << "input end" << std::endl;

  // The easy part: values at vertice 0,4,8,12
  for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
    {
      nodal_values[i * dofs_per_vertex] =
        support_point_values[i * dofs_per_vertex](0);
      std::cout << i * dofs_per_vertex << ": "
                << support_point_values[i * dofs_per_vertex](0) << std::endl;
    }
  if (dim == 2)
    {
      // First order derivatives at vertices 1,2,5,6,9,10,13,14
      nodal_values[1] = nodal_values[dofs_per_vertex] - nodal_values[0];
      nodal_values[2] = nodal_values[2 * dofs_per_vertex] - nodal_values[0];
      nodal_values[dofs_per_vertex + 1] = nodal_values[1];
      nodal_values[dofs_per_vertex + 2] =
        nodal_values[3 * dofs_per_vertex] - nodal_values[dofs_per_vertex];
      nodal_values[2 * dofs_per_vertex + 1] =
        nodal_values[3 * dofs_per_vertex] - nodal_values[2 * dofs_per_vertex];
      nodal_values[2 * dofs_per_vertex + 2] = nodal_values[2];
      nodal_values[3 * dofs_per_vertex + 1] =
        nodal_values[2 * dofs_per_vertex + 1];
      nodal_values[3 * dofs_per_vertex + 2] = nodal_values[dofs_per_vertex + 2];

      // Second order derivatives at vertices 3,7,11,15
      nodal_values[dim + 1] =
        nodal_values[2 * dofs_per_vertex + 1] - nodal_values[1];
      nodal_values[dofs_per_vertex + dim + 1] =
        nodal_values[3 * dofs_per_vertex + 1] -
        nodal_values[dofs_per_vertex + 1];
      nodal_values[2 * dofs_per_vertex + dim + 1] = nodal_values[dim + 1];
      nodal_values[3 * dofs_per_vertex + dim + 1] =
        nodal_values[2 * dofs_per_vertex + dim + 1];
    }
  else if (dim == 3)
    {
      // First order derivatives at vertices 1,2,4
      nodal_values[1] = nodal_values[dofs_per_vertex] - nodal_values[0];
      nodal_values[dofs_per_vertex + 1] = nodal_values[1];
      nodal_values[2 * dofs_per_vertex + 1] =
        nodal_values[3 * dofs_per_vertex] - nodal_values[2 * dofs_per_vertex];
      nodal_values[3 * dofs_per_vertex + 1] =
        nodal_values[2 * dofs_per_vertex + 1];
      nodal_values[4 * dofs_per_vertex + 1] =
        nodal_values[5 * dofs_per_vertex] - nodal_values[4 * dofs_per_vertex];
      nodal_values[5 * dofs_per_vertex + 1] =
        nodal_values[4 * dofs_per_vertex + 1];
      nodal_values[6 * dofs_per_vertex + 1] =
        nodal_values[7 * dofs_per_vertex] - nodal_values[6 * dofs_per_vertex];
      nodal_values[7 * dofs_per_vertex + 1] =
        nodal_values[6 * dofs_per_vertex + 1];

      nodal_values[2] = nodal_values[2 * dofs_per_vertex] - nodal_values[0];
      nodal_values[dofs_per_vertex + 2] =
        nodal_values[3 * dofs_per_vertex] - nodal_values[dofs_per_vertex];
      nodal_values[2 * dofs_per_vertex + 2] = nodal_values[2];
      nodal_values[3 * dofs_per_vertex + 2] = nodal_values[dofs_per_vertex + 2];
      nodal_values[4 * dofs_per_vertex + 2] =
        nodal_values[6 * dofs_per_vertex] - nodal_values[4 * dofs_per_vertex];
      nodal_values[5 * dofs_per_vertex + 2] =
        nodal_values[7 * dofs_per_vertex] - nodal_values[5 * dofs_per_vertex];
      nodal_values[6 * dofs_per_vertex + 2] =
        nodal_values[4 * dofs_per_vertex + 2];
      nodal_values[7 * dofs_per_vertex + 2] =
        nodal_values[5 * dofs_per_vertex + 2];

      nodal_values[4] = nodal_values[4 * dofs_per_vertex] - nodal_values[0];
      nodal_values[dofs_per_vertex + 4] =
        nodal_values[5 * dofs_per_vertex] - nodal_values[dofs_per_vertex];
      nodal_values[2 * dofs_per_vertex + 4] =
        nodal_values[6 * dofs_per_vertex] - nodal_values[2 * dofs_per_vertex];
      nodal_values[3 * dofs_per_vertex + 4] =
        nodal_values[7 * dofs_per_vertex] - nodal_values[3 * dofs_per_vertex];
      nodal_values[4 * dofs_per_vertex + 4] = nodal_values[4];
      nodal_values[5 * dofs_per_vertex + 4] = nodal_values[dofs_per_vertex + 4];
      nodal_values[6 * dofs_per_vertex + 4] =
        nodal_values[2 * dofs_per_vertex + 4];
      nodal_values[7 * dofs_per_vertex + 4] =
        nodal_values[3 * dofs_per_vertex + 4];

      // Second order derivatives at vertices 3,5,6
      // xy; y-derivative of the x-derivative
      nodal_values[3] = nodal_values[2 * dofs_per_vertex + 1] - nodal_values[1];
      nodal_values[dofs_per_vertex + 3] =
        nodal_values[3 * dofs_per_vertex + 1] -
        nodal_values[dofs_per_vertex + 1];
      nodal_values[2 * dofs_per_vertex + 3] = nodal_values[3];
      nodal_values[3 * dofs_per_vertex + 3] = nodal_values[dofs_per_vertex + 3];
      nodal_values[4 * dofs_per_vertex + 3] =
        nodal_values[6 * dofs_per_vertex + 1] -
        nodal_values[4 * dofs_per_vertex + 1];
      nodal_values[5 * dofs_per_vertex + 3] =
        nodal_values[7 * dofs_per_vertex + 1] -
        nodal_values[5 * dofs_per_vertex + 1];
      nodal_values[6 * dofs_per_vertex + 3] =
        nodal_values[4 * dofs_per_vertex + 3];
      nodal_values[7 * dofs_per_vertex + 3] =
        nodal_values[5 * dofs_per_vertex + 3];

      // xz; z-derivative of the x-derivative
      nodal_values[5] = nodal_values[4 * dofs_per_vertex + 1] - nodal_values[1];
      nodal_values[dofs_per_vertex + 5] =
        nodal_values[5 * dofs_per_vertex + 1] -
        nodal_values[dofs_per_vertex + 1];
      nodal_values[2 * dofs_per_vertex + 5] =
        nodal_values[6 * dofs_per_vertex + 1] -
        nodal_values[2 * dofs_per_vertex + 1];
      nodal_values[3 * dofs_per_vertex + 5] =
        nodal_values[7 * dofs_per_vertex + 1] -
        nodal_values[3 * dofs_per_vertex + 1];
      nodal_values[4 * dofs_per_vertex + 5] = nodal_values[5];
      nodal_values[5 * dofs_per_vertex + 5] = nodal_values[dofs_per_vertex + 5];
      nodal_values[6 * dofs_per_vertex + 5] =
        nodal_values[2 * dofs_per_vertex + 5];
      nodal_values[7 * dofs_per_vertex + 5] =
        nodal_values[3 * dofs_per_vertex + 5];

      // yz; z-derivative of the y-derivative
      nodal_values[6] = nodal_values[4 * dofs_per_vertex + 2] - nodal_values[2];
      nodal_values[dofs_per_vertex + 6] =
        nodal_values[5 * dofs_per_vertex + 2] -
        nodal_values[dofs_per_vertex + 2];
      nodal_values[2 * dofs_per_vertex + 6] =
        nodal_values[6 * dofs_per_vertex + 2] -
        nodal_values[2 * dofs_per_vertex + 2];
      nodal_values[3 * dofs_per_vertex + 6] =
        nodal_values[7 * dofs_per_vertex + 2] -
        nodal_values[3 * dofs_per_vertex + 2];
      nodal_values[4 * dofs_per_vertex + 6] = nodal_values[6];
      nodal_values[5 * dofs_per_vertex + 6] = nodal_values[dofs_per_vertex + 6];
      nodal_values[6 * dofs_per_vertex + 6] =
        nodal_values[2 * dofs_per_vertex + 6];
      nodal_values[7 * dofs_per_vertex + 6] =
        nodal_values[3 * dofs_per_vertex + 6];

      // Third order derivative at vertices 7,15,23,31,39,47,55,63
      // z-derivative of the xy-derivative
      nodal_values[7] = nodal_values[4 * dofs_per_vertex + 3] - nodal_values[3];
      nodal_values[dofs_per_vertex + 7] =
        nodal_values[5 * dofs_per_vertex + 3] -
        nodal_values[dofs_per_vertex + 3];
      nodal_values[2 * dofs_per_vertex + 7] =
        nodal_values[6 * dofs_per_vertex + 3] -
        nodal_values[2 * dofs_per_vertex + 3];
      nodal_values[3 * dofs_per_vertex + 7] =
        nodal_values[7 * dofs_per_vertex + 3] -
        nodal_values[3 * dofs_per_vertex + 3];
      nodal_values[4 * dofs_per_vertex + 7] = nodal_values[7];
      nodal_values[5 * dofs_per_vertex + 7] = nodal_values[dofs_per_vertex + 7];
      nodal_values[6 * dofs_per_vertex + 7] =
        nodal_values[2 * dofs_per_vertex + 7];
      nodal_values[7 * dofs_per_vertex + 7] =
        nodal_values[3 * dofs_per_vertex + 7];
    }

  // Assert(this->degree == 3, ExcNotImplemented());

  if (this->degree > 3)
    {
      // lines (4*2^1*(degree-3)^1 in 2d, 12*2^2*(degree-3)^1 in 3d
      const unsigned int starting_dof_lines =
        Utilities::pow(2, dim) * GeometryInfo<dim>::vertices_per_cell;
      const unsigned int dofs_per_line_vertex = Utilities::pow(2, dim - 1);
      {
        for (unsigned int line = 0; line < GeometryInfo<dim>::lines_per_cell;
             ++line)
          for (unsigned int i = 0; i < dofs_per_line; i += dofs_per_line_vertex)
            {
              nodal_values[starting_dof_lines + line * dofs_per_line + i] =
                support_point_values[starting_dof_lines + line * dofs_per_line +
                                     i](0);
            }

        if (dim == 2)
          {
            for (unsigned int i = 0; i < dofs_per_line;
                 i += dofs_per_line_vertex)
              {
                nodal_values[starting_dof_lines + i + 1] =
                  nodal_values[starting_dof_lines + dofs_per_line + i] -
                  nodal_values[starting_dof_lines + i];
                nodal_values[starting_dof_lines + dofs_per_line + i + 1] =
                  nodal_values[starting_dof_lines + i + 1];
                nodal_values[starting_dof_lines + 2 * dofs_per_line + i + 1] =
                  nodal_values[starting_dof_lines + 3 * dofs_per_line + i] -
                  nodal_values[starting_dof_lines + 2 * dofs_per_line + i];
                nodal_values[starting_dof_lines + 3 * dofs_per_line + i + 1] =
                  nodal_values[starting_dof_lines + 2 * dofs_per_line + i + 1];
              }
          }
        else
          {}
      }

      // faces 1*2^0*(degree-3)^2 in 2d, 6*2^1*(degree-3)^2 in 2d
      const unsigned int starting_dof_quads =
        starting_dof_lines + Utilities::pow(2, dim - 1) * (this->degree - 3) *
                               GeometryInfo<dim>::lines_per_cell;
      const unsigned int dofs_per_quad_vertex = Utilities::pow(2, dim - 2);
      {
        for (unsigned int quad = 0; quad < GeometryInfo<dim>::quads_per_cell;
             ++quad)
          for (unsigned int i = 0; i < dofs_per_quad; i += dofs_per_quad_vertex)
            nodal_values[starting_dof_quads + quad * dofs_per_quad_vertex + i] =
              support_point_values[starting_dof_quads +
                                   quad * dofs_per_quad_vertex + i](0);
      }

      if (dim >= 3)
        {
          // cell 1*2^0*(degree-3)^3 in 3d
          const unsigned int starting_dof_hexes =
            starting_dof_quads + Utilities::pow(2, dim - 2) *
                                   Utilities::pow(this->degree - 3, 2) *
                                   GeometryInfo<dim>::quads_per_cell;
          nodal_values[starting_dof_hexes] =
            support_point_values[starting_dof_hexes](0);

          AssertDimension((starting_dof_hexes +
                           Utilities::pow(this->degree - 3, 3)),
                          Utilities::pow(this->degree + 1, dim));
        }
    }
}

template <int dim, int spacedim>
bool
FE_Hermite<dim, spacedim>::hp_constraints_are_implemented() const
{
  return true;
}

template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_Hermite<dim, spacedim>::hp_vertex_dof_identities(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  // we can presently only compute these identities if both FEs are FE_Hermites
  // or if the other one is an FE_Nothing. in the first case, there should be
  // exactly one single DoF of each FE at a vertex, and they should have
  // identical value
  if (dynamic_cast<const FE_Hermite<dim, spacedim> *>(&fe_other) != nullptr)
    {
      return std::vector<std::pair<unsigned int, unsigned int>>(
        1, std::make_pair(0U, 0U));
    }
  else if (dynamic_cast<const FE_Nothing<dim> *>(&fe_other) != nullptr)
    {
      // the FE_Nothing has no degrees of freedom, so there are no
      // equivalencies to be recorded
      return std::vector<std::pair<unsigned int, unsigned int>>();
    }
  else if (fe_other.dofs_per_face == 0)
    {
      // if the other element has no elements on faces at all,
      // then it would be impossible to enforce any kind of
      // continuity even if we knew exactly what kind of element
      // we have -- simply because the other element declares
      // that it is discontinuous because it has no DoFs on
      // its faces. in that case, just state that we have no
      // constraints to declare
      return std::vector<std::pair<unsigned int, unsigned int>>();
    }
  else
    {
      Assert(false, ExcNotImplemented());
      return std::vector<std::pair<unsigned int, unsigned int>>();
    }
}

template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_Hermite<dim, spacedim>::hp_line_dof_identities(
  const FiniteElement<dim, spacedim> &) const
{
  // Since this fe is not interpolatory but on the vertices, we can
  // not identify dofs on lines and on quads even if there are dofs
  // on lines and on quads.
  //
  // we also have nothing to say about interpolation to other finite
  // elements. consequently, we never have anything to say at all
  return std::vector<std::pair<unsigned int, unsigned int>>();
}

template <int dim, int spacedim>
std::vector<std::pair<unsigned int, unsigned int>>
FE_Hermite<dim, spacedim>::hp_quad_dof_identities(
  const FiniteElement<dim, spacedim> &) const
{
  // Since this fe is not interpolatory but on the vertices, we can
  // not identify dofs on lines and on quads even if there are dofs
  // on lines and on quads.
  //
  // we also have nothing to say about interpolation to other finite
  // elements. consequently, we never have anything to say at all
  return std::vector<std::pair<unsigned int, unsigned int>>();
}

template <int dim, int spacedim>
FiniteElementDomination::Domination
FE_Hermite<dim, spacedim>::compare_for_face_domination(
  const FiniteElement<dim, spacedim> &fe_other) const
{
  if (const FE_Hermite<dim, spacedim> *fe_b_other =
        dynamic_cast<const FE_Hermite<dim, spacedim> *>(&fe_other))
    {
      if (this->degree < fe_b_other->degree)
        return FiniteElementDomination::this_element_dominates;
      else if (this->degree == fe_b_other->degree)
        return FiniteElementDomination::either_element_can_dominate;
      else
        return FiniteElementDomination::other_element_dominates;
    }
  else if (const FE_Nothing<dim> *fe_nothing =
             dynamic_cast<const FE_Nothing<dim> *>(&fe_other))
    {
      if (fe_nothing->is_dominating())
        {
          return FiniteElementDomination::other_element_dominates;
        }
      else
        {
          // the FE_Nothing has no degrees of freedom and it is typically used
          // in a context where we don't require any continuity along the
          // interface
          return FiniteElementDomination::no_requirements;
        }
    }

  Assert(false, ExcNotImplemented());
  return FiniteElementDomination::neither_element_dominates;
}

template <int dim, int spacedim>
std::string
FE_Hermite<dim, spacedim>::get_name() const
{
  // note that the FETools::get_fe_by_name function depends on the
  // particular format of the string this function returns, so they have to be
  // kept in synch

  std::ostringstream namebuf;
  namebuf << "FE_Hermite<" << dim << ">(" << this->degree << ")";
  return namebuf.str();
}

template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_Hermite<dim, spacedim>::clone() const
{
  return std_cxx14::make_unique<FE_Hermite<dim, spacedim>>(*this);
}

template <int dim, int spacedim>
std::vector<unsigned int>
FE_Hermite<dim, spacedim>::get_continuous_dpo_vector(const unsigned int deg)
{
  AssertThrow(deg > 0, ExcMessage("FE_Hermite needs to be of degree > 2."));
  std::vector<unsigned int> dpo(dim + 1, 0);
  for (unsigned int i = 0; i < dpo.size(); ++i)
    dpo[i] = Utilities::pow(2, dim - i) * Utilities::pow(deg - 3, i);

  return dpo;
}

template <int dim, int spacedim>
std::vector<unsigned int>
FE_Hermite<dim, spacedim>::get_dpo_vector(const unsigned int deg)
{
  AssertThrow(deg > 0, ExcMessage("FE_Hermite needs to be of degree > 2."));
  std::vector<unsigned int> dpo(dim + 1, 0);

  dpo[dim] = Utilities::pow(deg + 1, dim);

  return dpo;
}

template <int dim, int spacedim>
TensorProductPolynomials<dim>
FE_Hermite<dim, spacedim>::get_polynomials(const unsigned int degree)
{
  TensorProductPolynomials<dim> tpp(
    PolynomialsHermite<double>::generate_complete_basis(degree));

  const std::vector<unsigned int> renumbering =
    FETools::general_lexicographic_to_hierarchic<dim>(
      get_continuous_dpo_vector(degree));
  tpp.set_numbering(renumbering);

  return tpp;
}

/*template <int dim, int spacedim>
double
FE_Hermite<dim, spacedim>::evaluate_dof_for_shape_function(
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
}*/

template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::make_hanging_node_constraints(
  const DoFHandler<dim, spacedim> &dof_handler,
  AffineConstraints<double> &      constraints)
{
  FE_HermiteContinuous<dim, spacedim> fe_continuous(
    dof_handler.get_fe().degree);
  DoFHandler<dim, spacedim> dof_handler_continuous;
  dof_handler_continuous.initialize(dof_handler.get_triangulation(),
                                    fe_continuous);
  Assert(fe_continuous.degree == 3, ExcNotImplemented());

  // generate a point on this cell and evaluate the shape functions there
  const Quadrature<dim - 1> quad_face_support(
    fe_continuous.get_unit_face_support_points());

  std::vector<types::global_dof_index> dof_indices_own(
    dof_handler.get_fe().dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices_neighbor(
    dof_handler.get_fe().dofs_per_cell);

  FESubfaceValues<dim, spacedim> fe_values_own(fe_continuous,
                                               quad_face_support,
                                               update_quadrature_points |
                                                 update_values |
                                                 update_gradients |
                                                 update_hessians);
  FEFaceValues<dim, spacedim>    fe_values_neighbor(fe_continuous,
                                                 quad_face_support,
                                                 update_quadrature_points |
                                                   update_values |
                                                   update_gradients |
                                                   update_hessians);

  // Rule of thumb for FP accuracy, that can be expected for a given
  // polynomial degree. This value is used to cut off values close to
  // zero.
  // double eps = 2e-13 * fe_continuous.degree * (dim - 1);

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

                  const auto subface_cell_dg =
                    cell_dg->neighbor_child_on_subface(face_no, subface_no);
                  subface_cell_dg->get_dof_indices(dof_indices_neighbor);
                  for (unsigned int i = 0; i < fe_continuous.dofs_per_face;
                       i += fe_continuous.dofs_per_vertex)
                    {
                      for (unsigned int c = 0;
                           c < fe_continuous.dofs_per_vertex;
                           ++c)
                        Assert((fe_values_own.quadrature_point(i + c) -
                                fe_values_neighbor.quadrature_point(i + c))
                                   .norm() < 1.e-6,
                               ExcInternalError());
                      const auto first_neighbor_cell_index =
                        fe_continuous.face_to_cell_index(
                          i, cell_cg->neighbor_of_neighbor(face_no));

                      if (!constraints.is_constrained(
                            dof_indices_neighbor[first_neighbor_cell_index]))
                        {
                          for (unsigned int i = 0;
                               i < dof_handler.get_fe().n_components();
                               ++i)
                            for (unsigned int c = 0;
                                 c < fe_continuous.dofs_per_vertex;
                                 ++c)
                              constraints.add_line(
                                dof_indices_neighbor
                                  [first_neighbor_cell_index + c +
                                   i * fe_continuous.dofs_per_cell]);

                          double constrained_value =
                            fe_values_neighbor.shape_value(
                              first_neighbor_cell_index, i);
                          FullMatrix<double> inverted_constrained_gradients(
                            dim, dim);
                          {
                            FullMatrix<double> constrained_gradients(dim, dim);
                            if (dim == 2)
                              {
                                constrained_gradients[0][0] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 1, i + 1)[0];
                                constrained_gradients[1][0] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 1, i + 1)[1];
                                constrained_gradients[0][1] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 2, i + 2)[0];
                                constrained_gradients[1][1] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 2, i + 2)[1];
                              }
                            else
                              {
                                constrained_gradients[0][0] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 1, i + 1)[0];
                                constrained_gradients[1][0] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 1, i + 1)[1];
                                constrained_gradients[2][0] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 1, i + 1)[2];
                                constrained_gradients[0][1] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 2, i + 2)[0];
                                constrained_gradients[1][1] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 2, i + 2)[1];
                                constrained_gradients[2][1] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 2, i + 2)[2];
                                constrained_gradients[0][2] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 4, i + 4)[0];
                                constrained_gradients[1][2] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 4, i + 4)[1];
                                constrained_gradients[2][2] =
                                  fe_values_neighbor.shape_grad(
                                    first_neighbor_cell_index + 4, i + 4)[2];
                              }
                            inverted_constrained_gradients.invert(
                              constrained_gradients);
                          }

                          double constrained_2nd_derivative =
                            fe_values_neighbor.shape_hessian(
                              first_neighbor_cell_index + 3, i + 3)[0][1];

                          for (unsigned int j = 0;
                               j < fe_continuous.dofs_per_face;
                               ++j)
                            {
                              const auto own_cell_index =
                                fe_continuous.face_to_cell_index(j, face_no);

                              double constraining_value =
                                fe_values_own.shape_value(own_cell_index, i);
                              for (unsigned int c = 0;
                                   c < dof_handler.get_fe().n_components();
                                   ++c)
                                constraints.add_entry(
                                  dof_indices_neighbor
                                    [first_neighbor_cell_index +
                                     c * fe_continuous.dofs_per_cell],
                                  dof_indices_own[own_cell_index +
                                                  c * fe_continuous
                                                        .dofs_per_cell],
                                  constraining_value / constrained_value);

                              FullMatrix<double> constraining_gradients(dim, 1);
                              if (dim == 2)
                                {
                                  constraining_gradients[0][0] =
                                    fe_values_own.shape_grad(own_cell_index,
                                                             i)[0];
                                  constraining_gradients[1][0] =
                                    fe_values_own.shape_grad(own_cell_index,
                                                             i)[1];
                                }
                              else
                                {
                                  constraining_gradients[0][0] =
                                    fe_values_own.shape_grad(own_cell_index,
                                                             i)[0];
                                  constraining_gradients[1][0] =
                                    fe_values_own.shape_grad(own_cell_index,
                                                             i)[1];
                                  constraining_gradients[2][0] =
                                    fe_values_own.shape_grad(own_cell_index,
                                                             i)[2];
                                }
                              FullMatrix<double> constraint_matrix(dim, 1);
                              inverted_constrained_gradients.mmult(
                                constraint_matrix, constraining_gradients);
                              for (unsigned int c1 = 0; c1 < dim; ++c1)
                                for (unsigned int c = 0;
                                     c < dof_handler.get_fe().n_components();
                                     ++c)
                                  constraints.add_entry(
                                    dof_indices_neighbor
                                      [first_neighbor_cell_index +
                                       c * fe_continuous.dofs_per_cell +
                                       std::pow(2, c1)],
                                    dof_indices_own[own_cell_index +
                                                    c * fe_continuous
                                                          .dofs_per_cell],
                                    constraint_matrix[c1][0]);

                              double constraining_2nd_derivative =
                                fe_values_own.shape_hessian(own_cell_index,
                                                            i + 3)[0][1];
                              for (unsigned int c = 0;
                                   c < dof_handler.get_fe().n_components();
                                   ++c)
                                constraints.add_entry(
                                  dof_indices_neighbor
                                    [first_neighbor_cell_index +
                                     c * fe_continuous.dofs_per_cell + 3],
                                  dof_indices_own[own_cell_index +
                                                  c * fe_continuous
                                                        .dofs_per_cell],
                                  constraining_2nd_derivative /
                                    constrained_2nd_derivative);
                            }
                        }
                    }
                }
            }
        }
    }
}


template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::make_continuity_constraints(
  const DoFHandler<dim, spacedim> &dof_handler,
  AffineConstraints<double> &      constraints)
{
  const FiniteElement<dim, spacedim> &fe            = dof_handler.get_fe();
  const unsigned int                  dofs_per_cell = fe.dofs_per_cell;

  const Quadrature<dim> support(fe.get_unit_support_points());

  FEValues<dim, spacedim> fe_values_own(fe,
                                        support,
                                        update_quadrature_points |
                                          update_values | update_gradients |
                                          update_hessians);
  FEValues<dim, spacedim> fe_values_neighbor(fe,
                                             support,
                                             update_quadrature_points |
                                               update_values |
                                               update_gradients |
                                               update_hessians);

  std::vector<types::global_dof_index> dof_indices_own(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices_neighbor(dofs_per_cell);

  std::vector<Point<spacedim>> points_own;
  std::vector<Point<spacedim>> points_neighbor;

  std::vector<
    std::set<typename DoFHandler<dim, spacedim>::active_cell_iterator>>
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

        auto neighbor_cell_it = cell_it;
        ++neighbor_cell_it;
        for (; neighbor_cell_it != cells.end(); ++neighbor_cell_it)
          {
            const auto &neighbor_cell = *neighbor_cell_it;
            std::cout << "Neighbor center: " << neighbor_cell->center()
                      << std::endl;
            neighbor_cell->get_dof_indices(dof_indices_neighbor);
            {
              fe_values_neighbor.reinit(neighbor_cell);
              points_neighbor = fe_values_neighbor.get_quadrature_points();

              for (unsigned int i = 0; i < dofs_per_cell;
                   i += Utilities::pow(2, dim))
                {
                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                  for (unsigned int j = 0; j < dofs_per_cell;
                       j += Utilities::pow(2, dim))
                    {
                      const unsigned int component_j =
                        fe.system_to_component_index(j).first;
                      if ((points_own[i] - points_neighbor[j]).norm_square() <
                            1.e-12 &&
                          component_i == component_j)
                        {
                          const auto dof_indices_own_start = dof_indices_own[i];
                          const auto dof_indices_neighbor_start =
                            dof_indices_neighbor[j];
                          if (!constraints.is_constrained(
                                dof_indices_neighbor_start))
                            {
                              constraints.add_line(dof_indices_neighbor_start);
                              constraints.add_entry(dof_indices_neighbor_start,
                                                    dof_indices_own_start,
                                                    1.);
                            }
                          if (dim == 2)
                            {
                              FullMatrix<double> own_gradient_evaluations(dim,
                                                                          dim);
                              const double       factor = std::pow(
                                2., cell->level() - neighbor_cell->level());

                              for (unsigned int k = 0; k < dim; ++k)
                                if (!constraints.is_constrained(
                                      dof_indices_neighbor_start + k + 1))
                                  {
                                    constraints.add_line(
                                      dof_indices_neighbor_start + k + 1);
                                    constraints.add_entry(
                                      dof_indices_neighbor_start + k + 1,
                                      dof_indices_own_start + k + 1,
                                      factor);
                                  }

                              if (!constraints.is_constrained(
                                    dof_indices_neighbor_start + 3))
                                {
                                  constraints.add_line(
                                    dof_indices_neighbor_start + 3);
                                  constraints.add_entry(
                                    dof_indices_neighbor_start + 3,
                                    dof_indices_own_start + 3,
                                    factor * factor);
                                }
                            }
                          else if (dim == 3)
                            {
                              const double factor = std::pow(
                                2., cell->level() - neighbor_cell->level());

                              for (unsigned int k : {1, 2, 4})
                                if (!constraints.is_constrained(
                                      dof_indices_neighbor_start + k))
                                  {
                                    constraints.add_line(
                                      dof_indices_neighbor_start + k);
                                    constraints.add_entry(
                                      dof_indices_neighbor_start + k,
                                      dof_indices_own_start + k,
                                      factor);
                                  }
                            }
                          else
                            AssertThrow(false, ExcNotImplemented());
                        }
                    }
                }
            }
          }
      }
}


// explicit instantiations
#include "fe_hermite.inst"

DEAL_II_NAMESPACE_CLOSE
