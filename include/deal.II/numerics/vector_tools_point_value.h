// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2019 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#ifndef dealii_vector_tools_point_value_h
#define dealii_vector_tools_point_value_h


#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

namespace VectorTools
{
  /**
   * @name Evaluation of functions and errors
   */
  //@{

  /**
   * Point error evaluation. Find the first cell containing the given point
   * and compute the difference of a (possibly vector-valued) finite element
   * function and a continuous function (with as many vector components as the
   * finite element) at this point.
   *
   * This is a wrapper function using a Q1-mapping for cell boundaries to call
   * the other point_difference() function.
   *
   * @note If the cell in which the point is found is not locally owned, an
   * exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  point_difference(
    const DoFHandler<dim, spacedim> &                          dof,
    const VectorType &                                         fe_function,
    const Function<spacedim, typename VectorType::value_type> &exact_solution,
    Vector<typename VectorType::value_type> &                  difference,
    const Point<spacedim> &                                    point);

  /**
   * Point error evaluation. Find the first cell containing the given point
   * and compute the difference of a (possibly vector-valued) finite element
   * function and a continuous function (with as many vector components as the
   * finite element) at this point.
   *
   * Compared with the other function of the same name, this function uses an
   * arbitrary mapping to evaluate the difference.
   *
   * @note If the cell in which the point is found is not locally owned, an
   * exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  point_difference(
    const Mapping<dim, spacedim> &                             mapping,
    const DoFHandler<dim, spacedim> &                          dof,
    const VectorType &                                         fe_function,
    const Function<spacedim, typename VectorType::value_type> &exact_solution,
    Vector<typename VectorType::value_type> &                  difference,
    const Point<spacedim> &                                    point);

  /**
   * Evaluate a possibly vector-valued finite element function defined by the
   * given DoFHandler and nodal vector @p fe_function at the given point @p
   * point, and return the (vector) value of this function through the last
   * argument.
   *
   * This function uses a $Q_1$-mapping for the cell the point is evaluated
   * in. If you need to evaluate using a different mapping (for example when
   * using curved boundaries), use the point_difference() function that takes
   * a mapping.
   *
   * This function is not particularly cheap. This is because it first
   * needs to find which cell a given point is in, then find the point
   * on the reference cell that matches the given evaluation point,
   * and then evaluate the shape functions there. You probably do not
   * want to use this function to evaluate the solution at <i>many</i>
   * points. For this kind of application, the FEFieldFunction class
   * offers at least some optimizations. On the other hand, if you
   * want to evaluate <i>many solutions</i> at the same point, you may
   * want to look at the VectorTools::create_point_source_vector()
   * function.
   *
   * @note If the cell in which the point is found is not locally owned, an
   *   exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   *
   * @note This function needs to find the cell within which a point lies,
   *   and this can only be done up to a certain numerical tolerance of course.
   *   Consequently, for points that are on, or close to, the boundary of
   *   a cell, you may get the value of the finite element field either
   *   here or there, depending on which cell the point is found in. This
   *   does not matter (to within the same tolerance) if the finite element
   *   field is continuous. On the other hand, if the finite element in use
   *   is <i>not</i> continuous, then you will get unpredictable values for
   *   points on or close to the boundary of the cell, as one would expect
   *   when trying to evaluate point values of discontinuous functions.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  point_value(const DoFHandler<dim, spacedim> &        dof,
              const VectorType &                       fe_function,
              const Point<spacedim> &                  point,
              Vector<typename VectorType::value_type> &value);

  /**
   * Same as above for hp.
   *
   * @note If the cell in which the point is found is not locally owned, an
   * exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   *
   * @note This function needs to find the cell within which a point lies,
   *   and this can only be done up to a certain numerical tolerance of course.
   *   Consequently, for points that are on, or close to, the boundary of
   *   a cell, you may get the value of the finite element field either
   *   here or there, depending on which cell the point is found in. This
   *   does not matter (to within the same tolerance) if the finite element
   *   field is continuous. On the other hand, if the finite element in use
   *   is <i>not</i> continuous, then you will get unpredictable values for
   *   points on or close to the boundary of the cell, as one would expect
   *   when trying to evaluate point values of discontinuous functions.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  point_value(const hp::DoFHandler<dim, spacedim> &    dof,
              const VectorType &                       fe_function,
              const Point<spacedim> &                  point,
              Vector<typename VectorType::value_type> &value);

  /**
   * Evaluate a scalar finite element function defined by the given DoFHandler
   * and nodal vector @p fe_function at the given point @p point, and return
   * the value of this function.
   *
   * This function uses a Q1-mapping for the cell the point is evaluated
   * in. If you need to evaluate using a different mapping (for example when
   * using curved boundaries), use the point_difference() function that takes
   * a mapping.
   *
   * This function is not particularly cheap. This is because it first
   * needs to find which cell a given point is in, then find the point
   * on the reference cell that matches the given evaluation point,
   * and then evaluate the shape functions there. You probably do not
   * want to use this function to evaluate the solution at <i>many</i>
   * points. For this kind of application, the FEFieldFunction class
   * offers at least some optimizations. On the other hand, if you
   * want to evaluate <i>many solutions</i> at the same point, you may
   * want to look at the VectorTools::create_point_source_vector()
   * function.
   *
   * This function is used in the "Possibilities for extensions" part of the
   * results section of
   * @ref step_3 "step-3".
   *
   * @note If the cell in which the point is found is not locally owned, an
   * exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   *
   * @note This function needs to find the cell within which a point lies,
   *   and this can only be done up to a certain numerical tolerance of course.
   *   Consequently, for points that are on, or close to, the boundary of
   *   a cell, you may get the value of the finite element field either
   *   here or there, depending on which cell the point is found in. This
   *   does not matter (to within the same tolerance) if the finite element
   *   field is continuous. On the other hand, if the finite element in use
   *   is <i>not</i> continuous, then you will get unpredictable values for
   *   points on or close to the boundary of the cell, as one would expect
   *   when trying to evaluate point values of discontinuous functions.
   */
  template <int dim, typename VectorType, int spacedim>
  typename VectorType::value_type
  point_value(const DoFHandler<dim, spacedim> &dof,
              const VectorType &               fe_function,
              const Point<spacedim> &          point);

  /**
   * Same as above for hp.
   *
   * @note If the cell in which the point is found is not locally owned, an
   * exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   *
   * @note This function needs to find the cell within which a point lies,
   *   and this can only be done up to a certain numerical tolerance of course.
   *   Consequently, for points that are on, or close to, the boundary of
   *   a cell, you may get the value of the finite element field either
   *   here or there, depending on which cell the point is found in. This
   *   does not matter (to within the same tolerance) if the finite element
   *   field is continuous. On the other hand, if the finite element in use
   *   is <i>not</i> continuous, then you will get unpredictable values for
   *   points on or close to the boundary of the cell, as one would expect
   *   when trying to evaluate point values of discontinuous functions.
   */
  template <int dim, typename VectorType, int spacedim>
  typename VectorType::value_type
  point_value(const hp::DoFHandler<dim, spacedim> &dof,
              const VectorType &                   fe_function,
              const Point<spacedim> &              point);

  /**
   * Evaluate a possibly vector-valued finite element function defined by the
   * given DoFHandler and nodal vector @p fe_function at the given point @p
   * point, and return the (vector) value of this function through the last
   * argument.
   *
   * Compared with the other function of the same name, this function uses an
   * arbitrary mapping to evaluate the point value.
   *
   * This function is not particularly cheap. This is because it first
   * needs to find which cell a given point is in, then find the point
   * on the reference cell that matches the given evaluation point,
   * and then evaluate the shape functions there. You probably do not
   * want to use this function to evaluate the solution at <i>many</i>
   * points. For this kind of application, the FEFieldFunction class
   * offers at least some optimizations. On the other hand, if you
   * want to evaluate <i>many solutions</i> at the same point, you may
   * want to look at the VectorTools::create_point_source_vector()
   * function.
   *
   * @note If the cell in which the point is found is not locally owned, an
   * exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   *
   * @note This function needs to find the cell within which a point lies,
   *   and this can only be done up to a certain numerical tolerance of course.
   *   Consequently, for points that are on, or close to, the boundary of
   *   a cell, you may get the value of the finite element field either
   *   here or there, depending on which cell the point is found in. This
   *   does not matter (to within the same tolerance) if the finite element
   *   field is continuous. On the other hand, if the finite element in use
   *   is <i>not</i> continuous, then you will get unpredictable values for
   *   points on or close to the boundary of the cell, as one would expect
   *   when trying to evaluate point values of discontinuous functions.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  point_value(const Mapping<dim, spacedim> &           mapping,
              const DoFHandler<dim, spacedim> &        dof,
              const VectorType &                       fe_function,
              const Point<spacedim> &                  point,
              Vector<typename VectorType::value_type> &value);

  /**
   * Same as above for hp.
   *
   * @note If the cell in which the point is found is not locally owned, an
   * exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   *
   * @note This function needs to find the cell within which a point lies,
   *   and this can only be done up to a certain numerical tolerance of course.
   *   Consequently, for points that are on, or close to, the boundary of
   *   a cell, you may get the value of the finite element field either
   *   here or there, depending on which cell the point is found in. This
   *   does not matter (to within the same tolerance) if the finite element
   *   field is continuous. On the other hand, if the finite element in use
   *   is <i>not</i> continuous, then you will get unpredictable values for
   *   points on or close to the boundary of the cell, as one would expect
   *   when trying to evaluate point values of discontinuous functions.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  point_value(const hp::MappingCollection<dim, spacedim> &mapping,
              const hp::DoFHandler<dim, spacedim> &       dof,
              const VectorType &                          fe_function,
              const Point<spacedim> &                     point,
              Vector<typename VectorType::value_type> &   value);

  /**
   * Evaluate a scalar finite element function defined by the given DoFHandler
   * and nodal vector @p fe_function at the given point @p point, and return
   * the value of this function.
   *
   * Compared with the other function of the same name, this function uses an
   * arbitrary mapping to evaluate the difference.
   *
   * This function is not particularly cheap. This is because it first
   * needs to find which cell a given point is in, then find the point
   * on the reference cell that matches the given evaluation point,
   * and then evaluate the shape functions there. You probably do not
   * want to use this function to evaluate the solution at <i>many</i>
   * points. For this kind of application, the FEFieldFunction class
   * offers at least some optimizations. On the other hand, if you
   * want to evaluate <i>many solutions</i> at the same point, you may
   * want to look at the VectorTools::create_point_source_vector()
   * function.
   *
   * @note If the cell in which the point is found is not locally owned, an
   * exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   *
   * @note This function needs to find the cell within which a point lies,
   *   and this can only be done up to a certain numerical tolerance of course.
   *   Consequently, for points that are on, or close to, the boundary of
   *   a cell, you may get the value of the finite element field either
   *   here or there, depending on which cell the point is found in. This
   *   does not matter (to within the same tolerance) if the finite element
   *   field is continuous. On the other hand, if the finite element in use
   *   is <i>not</i> continuous, then you will get unpredictable values for
   *   points on or close to the boundary of the cell, as one would expect
   *   when trying to evaluate point values of discontinuous functions.
   */
  template <int dim, typename VectorType, int spacedim>
  typename VectorType::value_type
  point_value(const Mapping<dim, spacedim> &   mapping,
              const DoFHandler<dim, spacedim> &dof,
              const VectorType &               fe_function,
              const Point<spacedim> &          point);

  /**
   * Same as above for hp.
   *
   * @note If the cell in which the point is found is not locally owned, an
   * exception of type VectorTools::ExcPointNotAvailableHere is thrown.
   *
   * @note This function needs to find the cell within which a point lies,
   *   and this can only be done up to a certain numerical tolerance of course.
   *   Consequently, for points that are on, or close to, the boundary of
   *   a cell, you may get the value of the finite element field either
   *   here or there, depending on which cell the point is found in. This
   *   does not matter (to within the same tolerance) if the finite element
   *   field is continuous. On the other hand, if the finite element in use
   *   is <i>not</i> continuous, then you will get unpredictable values for
   *   points on or close to the boundary of the cell, as one would expect
   *   when trying to evaluate point values of discontinuous functions.
   */
  template <int dim, typename VectorType, int spacedim>
  typename VectorType::value_type
  point_value(const hp::MappingCollection<dim, spacedim> &mapping,
              const hp::DoFHandler<dim, spacedim> &       dof,
              const VectorType &                          fe_function,
              const Point<spacedim> &                     point);
  //@}
} // namespace VectorTools

DEAL_II_NAMESPACE_CLOSE

#endif // dealii_vector_tools_point_value_h