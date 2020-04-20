// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2018 by the deal.II authors
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

#ifndef dealii_vector_tools_rhs_h
#define dealii_vector_tools_rhs_h

#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

namespace VectorTools
{
  /**
   * @name Assembling of right hand sides
   */
  //@{

  /**
   * Create a right hand side vector from boundary forces. Prior content of
   * the given @p rhs_vector vector is deleted.
   *
   * See the general documentation of this namespace for further information.
   *
   * @see
   * @ref GlossBoundaryIndicator "Glossary entry on boundary indicators"
   */
  template <int dim, int spacedim, typename VectorType>
  void
  create_boundary_right_hand_side(
    const Mapping<dim, spacedim> &                             mapping,
    const DoFHandler<dim, spacedim> &                          dof,
    const Quadrature<dim - 1> &                                q,
    const Function<spacedim, typename VectorType::value_type> &rhs,
    VectorType &                                               rhs_vector,
    const std::set<types::boundary_id> &                       boundary_ids =
      std::set<types::boundary_id>());

  /**
   * Call the create_boundary_right_hand_side() function, see above, with
   * <tt>mapping=MappingQGeneric@<dim@>(1)</tt>.
   *
   * @see
   * @ref GlossBoundaryIndicator "Glossary entry on boundary indicators"
   */
  template <int dim, int spacedim, typename VectorType>
  void
  create_boundary_right_hand_side(
    const DoFHandler<dim, spacedim> &                          dof,
    const Quadrature<dim - 1> &                                q,
    const Function<spacedim, typename VectorType::value_type> &rhs,
    VectorType &                                               rhs_vector,
    const std::set<types::boundary_id> &                       boundary_ids =
      std::set<types::boundary_id>());

  /**
   * Same as the set of functions above, but for hp objects.
   *
   * @see
   * @ref GlossBoundaryIndicator "Glossary entry on boundary indicators"
   */
  template <int dim, int spacedim, typename VectorType>
  void
  create_boundary_right_hand_side(
    const hp::MappingCollection<dim, spacedim> &               mapping,
    const hp::DoFHandler<dim, spacedim> &                      dof,
    const hp::QCollection<dim - 1> &                           q,
    const Function<spacedim, typename VectorType::value_type> &rhs,
    VectorType &                                               rhs_vector,
    const std::set<types::boundary_id> &                       boundary_ids =
      std::set<types::boundary_id>());

  /**
   * Call the create_boundary_right_hand_side() function, see above, with a
   * single Q1 mapping as collection. This function therefore will only work
   * if the only active fe index in use is zero.
   *
   * @see
   * @ref GlossBoundaryIndicator "Glossary entry on boundary indicators"
   */
  template <int dim, int spacedim, typename VectorType>
  void
  create_boundary_right_hand_side(
    const hp::DoFHandler<dim, spacedim> &                      dof,
    const hp::QCollection<dim - 1> &                           q,
    const Function<spacedim, typename VectorType::value_type> &rhs,
    VectorType &                                               rhs_vector,
    const std::set<types::boundary_id> &                       boundary_ids =
      std::set<types::boundary_id>());
  // @}
} // namespace VectorTools

DEAL_II_NAMESPACE_CLOSE

#endif // dealii_vector_tools_rhs_h