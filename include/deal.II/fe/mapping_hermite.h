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

#ifndef dealii_mapping_hermite
#define dealii_mapping_hermite

#include <deal.II/base/config.h>

#include <deal.II/base/table.h>
#include <deal.II/base/thread_management.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <array>

DEAL_II_NAMESPACE_OPEN

/*!@addtogroup mapping */
/*@{*/

template <int dim, int spacedim>
class MappingHermiteHelper
{
public:
  MappingHermiteHelper(const Triangulation<dim, spacedim> &triangulation);

  void
  reinit();

  const LinearAlgebra::distributed::Vector<double> &
  get_hermite_vector() const;

  const DoFHandler<dim, spacedim> &
  get_dof_handler() const;

private:
  std::shared_ptr<DoFHandler<dim, spacedim>> dof_handler;
  LinearAlgebra::distributed::Vector<double> hermite_vector;
};

/**
 * The MappingHermite is a generalization of the MappingQEulerian class, for
 * arbitrary vector finite elements. The two main differences are that this
 * class uses a vector of absolute positions instead of a vector of
 * displacements, and it allows for arbitrary FiniteElement types, instead of
 * only FE_Q.
 *
 * This class effectively decouples the topology from the geometry, by
 * relegating all geometrical information to some components of a
 * FiniteElement vector field. The components that are used for the geometry
 * can be arbitrarily selected at construction time.
 *
 * The idea is to consider the Triangulation as a parameter configuration
 * space, on which we  construct an arbitrary geometrical mapping, using the
 * instruments of the deal.II library: a vector of degrees of freedom, a
 * DoFHandler associated to the geometry of the problem and a ComponentMask
 * that tells us which components of the FiniteElement to use for the mapping.
 *
 * Typically, the DoFHandler operates on a finite element that is constructed
 * as a system element (FESystem()) from continuous FE_Q() (for iso-parametric
 * discretizations) or FE_Bernstein() (for iso-geometric discretizations)
 * objects. An example is shown below:
 *
 * @code
 *    const FE_Q<dim,spacedim> feq(1);
 *    const FESystem<dim,spacedim> fesystem(feq, spacedim);
 *    DoFHandler<dim,spacedim> dhq(triangulation);
 *    dhq.distribute_dofs(fesystem);
 *    const ComponentMask mask(spacedim, true);
 *    Vector<double> eulerq(dhq.n_dofs());
 *    // Fills the euler vector with information from the Triangulation
 *    VectorTools::get_position_vector(dhq, eulerq, mask);
 *    MappingHermite<dim,spacedim> map(dhq, eulerq, mask);
 * @endcode
 *
 * @author Luca Heltai, Marco Tezzele 2013, 2015
 */
template <int dim, int spacedim = dim>
class MappingHermite
  : public MappingHermiteHelper<dim, spacedim>,
    public MappingFEField<dim,
                          spacedim,
                          LinearAlgebra::distributed::Vector<double>,
                          DoFHandler<dim, spacedim>>
{
public:
  /**
   * Constructor.
   */
  MappingHermite(const Triangulation<dim, spacedim> &triangulation);

  void
  reinit();

  /**
   * Return a pointer to a copy of the present object. The caller of this copy
   * then assumes ownership of it.
   */
  virtual std::unique_ptr<Mapping<dim, spacedim>>
  clone() const override;

  /**
   * See the documentation of Mapping::preserves_vertex_locations()
   * for the purpose of this function. The implementation in this
   * class always returns @p true.
   */
  virtual bool
  preserves_vertex_locations() const override;
};

/*@}*/

/* -------------- declaration of explicit specializations ------------- */

#ifndef DOXYGEN

#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
