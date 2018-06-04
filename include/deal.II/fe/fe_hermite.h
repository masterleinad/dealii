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

#ifndef dealii_fe_hermite_h
#define dealii_fe_hermite_h

#include <deal.II/base/config.h>

#include <deal.II/base/tensor_product_polynomials.h>

#include <deal.II/fe/fe_q_base.h>

DEAL_II_NAMESPACE_OPEN

/*!@addtogroup fe */
/*@{*/

/**
 * Implementation of a scalar Hermite finite element @p that we call
 * FE_Hermite in analogy with FE_Q that yields the finite element space of
 * continuous, piecewise Hermite polynomials of degree @p p in each
 * coordinate direction. This class is realized using tensor product
 * polynomials of Hermite basis polynomials.
 *
 *
 * The standard constructor of this class takes the degree @p p of this finite
 * element.
 *
 * For more information about the <tt>spacedim</tt> template parameter check
 * the documentation of FiniteElement or the one of Triangulation.
 *
 * <h3>Implementation</h3>
 *
 * The constructor creates a TensorProductPolynomials object that includes the
 * tensor product of @p Hermite polynomials of degree @p p. This @p
 * TensorProductPolynomials object provides all values and derivatives of the
 * shape functions.
 *
 * <h3>Numbering of the degrees of freedom (DoFs)</h3>
 *
 * The original ordering of the shape functions represented by the
 * TensorProductPolynomials is a tensor product numbering. However, the shape
 * functions on a cell are renumbered beginning with the shape functions whose
 * support points are at the vertices, then on the line, on the quads, and
 * finally (for 3d) on the hexes. See the documentation of FE_Q for more
 * details.
 *
 *
 * @author Marco Tezzele, Luca Heltai
 * @date 2013, 2015
 */

template <int dim, int spacedim = dim>
class FE_Hermite : public FE_Poly<TensorProductPolynomials<dim>, dim, spacedim>
{
public:
  /**
   * Constructor for tensor product polynomials of degree @p p.
   */
  FE_Hermite(const unsigned int p);

  /**
   * FE_Hermite is not interpolatory in the element interior, which prevents
   * this element from defining an interpolation matrix. An exception will be
   * thrown.
   *
   * This function overrides the implementation from FE_Q_Base.
   */
  virtual void
  get_interpolation_matrix(const FiniteElement<dim, spacedim> &source,
                           FullMatrix<double> &matrix) const override;

  /**
   * FE_Hermite is not interpolatory in the element interior, which prevents
   * this element from defining a restriction matrix. An exception will be
   * thrown.
   *
   * This function overrides the implementation from FE_Q_Base.
   */
  virtual const FullMatrix<double> &
  get_restriction_matrix(
    const unsigned int         child,
    const RefinementCase<dim> &refinement_case =
      RefinementCase<dim>::isotropic_refinement) const override;

  /**
   * FE_Hermite is not interpolatory in the element interior, which prevents
   * this element from defining a prolongation matrix. An exception will be
   * thrown.
   *
   * This function overrides the implementation from FE_Q_Base.
   */
  virtual const FullMatrix<double> &
  get_prolongation_matrix(
    const unsigned int         child,
    const RefinementCase<dim> &refinement_case =
      RefinementCase<dim>::isotropic_refinement) const override;

  /**
   * Return the matrix interpolating from a face of one element to the face of
   * the neighboring element.  The size of the matrix is then
   * <tt>source.dofs_per_face</tt> times <tt>this->dofs_per_face</tt>. The
   * FE_Hermite element family only provides interpolation matrices for
   * elements of the same type and FE_Nothing. For all other elements, an
   * exception of type
   * FiniteElement<dim,spacedim>::ExcInterpolationNotImplemented is thrown.
   */
  virtual void
  get_face_interpolation_matrix(const FiniteElement<dim, spacedim> &source,
                                FullMatrix<double> &matrix) const override;

  /**
   * Return the matrix interpolating from a face of one element to the face of
   * the neighboring element.  The size of the matrix is then
   * <tt>source.dofs_per_face</tt> times <tt>this->dofs_per_face</tt>. The
   * FE_Hermite element family only provides interpolation matrices for
   * elements of the same type and FE_Nothing. For all other elements, an
   * exception of type
   * FiniteElement<dim,spacedim>::ExcInterpolationNotImplemented is thrown.
   */
  virtual void
  get_subface_interpolation_matrix(const FiniteElement<dim, spacedim> &source,
                                   const unsigned int                  subface,
                                   FullMatrix<double> &matrix) const override;

  virtual void
  convert_generalized_support_point_values_to_dof_values(
    const std::vector<Vector<double>> &support_point_values,
    std::vector<double> &              nodal_values) const override;

  /**
   * Return whether this element implements its hanging node constraints in
   * the new way, which has to be used to make elements "hp compatible".
   */
  virtual bool
  hp_constraints_are_implemented() const override;

  /**
   * If, on a vertex, several finite elements are active, the hp code first
   * assigns the degrees of freedom of each of these FEs different global
   * indices. It then calls this function to find out which of them should get
   * identical values, and consequently can receive the same global DoF index.
   * This function therefore returns a list of identities between DoFs of the
   * present finite element object with the DoFs of @p fe_other, which is a
   * reference to a finite element object representing one of the other finite
   * elements active on this particular vertex. The function computes which of
   * the degrees of freedom of the two finite element objects are equivalent,
   * both numbered between zero and the corresponding value of dofs_per_vertex
   * of the two finite elements. The first index of each pair denotes one of
   * the vertex dofs of the present element, whereas the second is the
   * corresponding index of the other finite element.
   */
  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_vertex_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  /**
   * Same as hp_vertex_dof_indices(), except that the function treats degrees
   * of freedom on lines.
   */
  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_line_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  /**
   * Same as hp_vertex_dof_indices(), except that the function treats degrees
   * of freedom on quads.
   */
  virtual std::vector<std::pair<unsigned int, unsigned int>>
  hp_quad_dof_identities(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  /**
   * Return whether this element dominates the one given as argument when they
   * meet at a common face, whether it is the other way around, whether
   * neither dominates, or if either could dominate.
   *
   * For a definition of domination, see FiniteElementDomination::Domination
   * and in particular the
   * @ref hp_paper "hp paper".
   */
  virtual FiniteElementDomination::Domination
  compare_for_face_domination(
    const FiniteElement<dim, spacedim> &fe_other) const override;

  /**
   * Return a string that uniquely identifies a finite element. This class
   * returns <tt>FE_Hermite<dim>(degree)</tt>, with @p dim and @p degree
   * replaced by appropriate values.
   */
  virtual std::string
  get_name() const override;

  virtual std::unique_ptr<FiniteElement<dim, spacedim>>
  clone() const override;

private:
  /**
   * Create the @p dofs_per_object vector that is needed within the constructor
   * to be passed to the constructor of @p FiniteElementData.
   */
  static std::vector<unsigned int>
  get_dpo_vector(const unsigned int degree);

  /**
   * Return the polynomial space used with a suitable renumbering.
   */
  TensorProductPolynomials<dim>
  get_polynomials(const unsigned int degree);

  /**
   * Initialize the @p unit_support_points variable.
   */
  void
  initialize_unit_support_points(const std::vector<Point<1>> &points);

  /**
   * Initialize the @p unit_face_support_points variable.
   */
  void
  initialize_unit_face_support_points(const std::vector<Point<1>> &points);
};

/*@}*/

DEAL_II_NAMESPACE_CLOSE

#endif
