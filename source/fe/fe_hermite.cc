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

#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

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
  /*std::vector<Point<1>> points_1d;
  points_1d.reserve(degree - 1);
  for (unsigned int i = 0; i < degree - 1; ++i)
    points_1d.push_back(Point<1>(i * 1. / (degree - 2)));
  initialize_unit_support_points(points_1d);
  initialize_unit_face_support_points(points_1d);*/
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
  const unsigned int dof,
  const Point<dim> & p,
  const unsigned int shape_function) const
{
  (void)dof;
  double scale = .5;
  if (dim == 2 && this->degree == 3)
    {
      switch (dof % 4)
        {
          case 0:
            return this->shape_value(shape_function, p);
          case 1:
            return this->shape_grad(shape_function, p)[0] * scale;
          case 2:
            return this->shape_grad(shape_function, p)[1] * scale;
            break;
          case 3:
            return this->shape_grad_grad(shape_function, p)[0][1] * scale *
                   scale;
          default:
            Assert(false, ExcInternalError());
        }
      return this->shape_value(shape_function, p);
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
          const Point<dim> &p = subface_quadrature.point(i);

          for (unsigned int j = 0; j < this->dofs_per_face; ++j)
            {
              double matrix_entry =
                evaluate_dof_for_shape_function(this->face_to_cell_index(i, 0),
                                                p,
                                                this->face_to_cell_index(j, 0));

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

  // The easy part: values at vertice 0,4,8,12
  for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
    {
      nodal_values[i * this->dofs_per_vertex] =
        support_point_values[i * this->dofs_per_vertex](0);
    }
  if (dim == 2)
    {
      // First order derivatives at vertices 1,2,5,6,9,10,13,14
      nodal_values[1] = nodal_values[this->dofs_per_vertex] - nodal_values[0];
      nodal_values[2] =
        nodal_values[2 * this->dofs_per_vertex] - nodal_values[0];
      nodal_values[this->dofs_per_vertex + 1] = nodal_values[1];
      nodal_values[this->dofs_per_vertex + 2] =
        nodal_values[3 * this->dofs_per_vertex] -
        nodal_values[this->dofs_per_vertex];
      nodal_values[2 * this->dofs_per_vertex + 1] =
        nodal_values[3 * this->dofs_per_vertex] -
        nodal_values[2 * this->dofs_per_vertex];
      nodal_values[2 * this->dofs_per_vertex + 2] = nodal_values[2];
      nodal_values[3 * this->dofs_per_vertex + 1] =
        nodal_values[2 * this->dofs_per_vertex + 1];
      nodal_values[3 * this->dofs_per_vertex + 2] =
        nodal_values[this->dofs_per_vertex + 2];

      // Second order derivatives at vertices 3,7,11,15
      nodal_values[dim + 1] =
        nodal_values[2 * this->dofs_per_vertex + 1] - nodal_values[1];
      nodal_values[this->dofs_per_vertex + dim + 1] =
        nodal_values[3 * this->dofs_per_vertex + 1] -
        nodal_values[this->dofs_per_vertex + 1];
      nodal_values[2 * this->dofs_per_vertex + dim + 1] = nodal_values[dim + 1];
      nodal_values[3 * this->dofs_per_vertex + dim + 1] =
        nodal_values[2 * this->dofs_per_vertex + dim + 1];
    }
  else if (dim == 3)
    {
      // First order derivatives at vertices 1,2,4
      nodal_values[1] = nodal_values[this->dofs_per_vertex] - nodal_values[0];
      nodal_values[this->dofs_per_vertex + 1] = nodal_values[1];
      nodal_values[2 * this->dofs_per_vertex + 1] =
        nodal_values[3 * this->dofs_per_vertex] -
        nodal_values[2 * this->dofs_per_vertex];
      nodal_values[3 * this->dofs_per_vertex + 1] =
        nodal_values[2 * this->dofs_per_vertex + 1];
      nodal_values[4 * this->dofs_per_vertex + 1] =
        nodal_values[5 * this->dofs_per_vertex] -
        nodal_values[4 * this->dofs_per_vertex];
      nodal_values[5 * this->dofs_per_vertex + 1] =
        nodal_values[4 * this->dofs_per_vertex + 1];
      nodal_values[6 * this->dofs_per_vertex + 1] =
        nodal_values[7 * this->dofs_per_vertex] -
        nodal_values[6 * this->dofs_per_vertex];
      nodal_values[7 * this->dofs_per_vertex + 1] =
        nodal_values[6 * this->dofs_per_vertex + 1];

      nodal_values[2] =
        nodal_values[2 * this->dofs_per_vertex] - nodal_values[0];
      nodal_values[this->dofs_per_vertex + 2] =
        nodal_values[3 * this->dofs_per_vertex] -
        nodal_values[this->dofs_per_vertex];
      nodal_values[2 * this->dofs_per_vertex + 2] = nodal_values[2];
      nodal_values[3 * this->dofs_per_vertex + 2] =
        nodal_values[this->dofs_per_vertex + 2];
      nodal_values[4 * this->dofs_per_vertex + 2] =
        nodal_values[6 * this->dofs_per_vertex] -
        nodal_values[4 * this->dofs_per_vertex];
      nodal_values[5 * this->dofs_per_vertex + 2] =
        nodal_values[7 * this->dofs_per_vertex] -
        nodal_values[5 * this->dofs_per_vertex];
      nodal_values[6 * this->dofs_per_vertex + 2] =
        nodal_values[4 * this->dofs_per_vertex + 2];
      nodal_values[7 * this->dofs_per_vertex + 2] =
        nodal_values[5 * this->dofs_per_vertex + 2];

      nodal_values[4] =
        nodal_values[4 * this->dofs_per_vertex] - nodal_values[0];
      nodal_values[this->dofs_per_vertex + 4] =
        nodal_values[5 * this->dofs_per_vertex] -
        nodal_values[this->dofs_per_vertex];
      nodal_values[2 * this->dofs_per_vertex + 4] =
        nodal_values[6 * this->dofs_per_vertex] -
        nodal_values[2 * this->dofs_per_vertex];
      nodal_values[3 * this->dofs_per_vertex + 4] =
        nodal_values[7 * this->dofs_per_vertex] -
        nodal_values[3 * this->dofs_per_vertex];
      nodal_values[4 * this->dofs_per_vertex + 4] = nodal_values[4];
      nodal_values[5 * this->dofs_per_vertex + 4] =
        nodal_values[this->dofs_per_vertex + 4];
      nodal_values[6 * this->dofs_per_vertex + 4] =
        nodal_values[2 * this->dofs_per_vertex + 4];
      nodal_values[7 * this->dofs_per_vertex + 4] =
        nodal_values[3 * this->dofs_per_vertex + 4];

      // Second order derivatives at vertices 3,5,6
      // xy; y-derivative of the x-derivative
      nodal_values[3] =
        nodal_values[2 * this->dofs_per_vertex + 1] - nodal_values[1];
      nodal_values[this->dofs_per_vertex + 3] =
        nodal_values[3 * this->dofs_per_vertex + 1] -
        nodal_values[this->dofs_per_vertex + 1];
      nodal_values[2 * this->dofs_per_vertex + 3] = nodal_values[3];
      nodal_values[3 * this->dofs_per_vertex + 3] =
        nodal_values[this->dofs_per_vertex + 3];
      nodal_values[4 * this->dofs_per_vertex + 3] =
        nodal_values[6 * this->dofs_per_vertex + 1] -
        nodal_values[4 * this->dofs_per_vertex + 1];
      nodal_values[5 * this->dofs_per_vertex + 3] =
        nodal_values[7 * this->dofs_per_vertex + 1] -
        nodal_values[5 * this->dofs_per_vertex + 1];
      nodal_values[6 * this->dofs_per_vertex + 3] =
        nodal_values[4 * this->dofs_per_vertex + 3];
      nodal_values[7 * this->dofs_per_vertex + 3] =
        nodal_values[5 * this->dofs_per_vertex + 3];

      // xz; z-derivative of the x-derivative
      nodal_values[5] =
        nodal_values[4 * this->dofs_per_vertex + 1] - nodal_values[1];
      nodal_values[this->dofs_per_vertex + 5] =
        nodal_values[5 * this->dofs_per_vertex + 1] -
        nodal_values[this->dofs_per_vertex + 1];
      nodal_values[2 * this->dofs_per_vertex + 5] =
        nodal_values[6 * this->dofs_per_vertex + 1] -
        nodal_values[2 * this->dofs_per_vertex + 1];
      nodal_values[3 * this->dofs_per_vertex + 5] =
        nodal_values[7 * this->dofs_per_vertex + 1] -
        nodal_values[3 * this->dofs_per_vertex + 1];
      nodal_values[4 * this->dofs_per_vertex + 5] = nodal_values[5];
      nodal_values[5 * this->dofs_per_vertex + 5] =
        nodal_values[this->dofs_per_vertex + 5];
      nodal_values[6 * this->dofs_per_vertex + 5] =
        nodal_values[2 * this->dofs_per_vertex + 5];
      nodal_values[7 * this->dofs_per_vertex + 5] =
        nodal_values[3 * this->dofs_per_vertex + 5];

      // yz; z-derivative of the y-derivative
      nodal_values[6] =
        nodal_values[4 * this->dofs_per_vertex + 2] - nodal_values[2];
      nodal_values[this->dofs_per_vertex + 6] =
        nodal_values[5 * this->dofs_per_vertex + 2] -
        nodal_values[this->dofs_per_vertex + 2];
      nodal_values[2 * this->dofs_per_vertex + 6] =
        nodal_values[6 * this->dofs_per_vertex + 2] -
        nodal_values[2 * this->dofs_per_vertex + 2];
      nodal_values[3 * this->dofs_per_vertex + 6] =
        nodal_values[7 * this->dofs_per_vertex + 2] -
        nodal_values[3 * this->dofs_per_vertex + 2];
      nodal_values[4 * this->dofs_per_vertex + 6] = nodal_values[6];
      nodal_values[5 * this->dofs_per_vertex + 6] =
        nodal_values[this->dofs_per_vertex + 6];
      nodal_values[6 * this->dofs_per_vertex + 6] =
        nodal_values[2 * this->dofs_per_vertex + 6];
      nodal_values[7 * this->dofs_per_vertex + 6] =
        nodal_values[3 * this->dofs_per_vertex + 6];

      // Third order derivative at vertices 7,15,23,31,39,47,55,63
      // z-derivative of the xy-derivative
      nodal_values[7] =
        nodal_values[4 * this->dofs_per_vertex + 3] - nodal_values[3];
      nodal_values[this->dofs_per_vertex + 7] =
        nodal_values[5 * this->dofs_per_vertex + 3] -
        nodal_values[this->dofs_per_vertex + 3];
      nodal_values[2 * this->dofs_per_vertex + 7] =
        nodal_values[6 * this->dofs_per_vertex + 3] -
        nodal_values[2 * this->dofs_per_vertex + 3];
      nodal_values[3 * this->dofs_per_vertex + 7] =
        nodal_values[7 * this->dofs_per_vertex + 3] -
        nodal_values[3 * this->dofs_per_vertex + 3];
      nodal_values[4 * this->dofs_per_vertex + 7] = nodal_values[7];
      nodal_values[5 * this->dofs_per_vertex + 7] =
        nodal_values[this->dofs_per_vertex + 7];
      nodal_values[6 * this->dofs_per_vertex + 7] =
        nodal_values[2 * this->dofs_per_vertex + 7];
      nodal_values[7 * this->dofs_per_vertex + 7] =
        nodal_values[3 * this->dofs_per_vertex + 7];
    }

  Assert(this->degree == 3, ExcNotImplemented());
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
  /*const std::vector<unsigned int> renumbering =
    FETools::general_lexicographic_to_hierarchic<dim>(
      this->get_dpo_vector(degree));
  tpp.set_numbering(renumbering);*/

  return tpp;
}

// explicit instantiations
#include "fe_hermite.inst"

DEAL_II_NAMESPACE_CLOSE
