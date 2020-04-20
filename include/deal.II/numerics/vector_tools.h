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

#ifndef dealii_vector_tools_h
#define dealii_vector_tools_h


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/deprecated_function_map.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/mapping_collection.h>

#include <deal.II/numerics/vector_tools_common.h>

#include <functional>
#include <map>
#include <set>
#include <vector>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
template <int dim, typename RangeNumberType>
class Function;
template <int dim>
class Quadrature;
template <int dim>
class QGauss;
template <int dim, typename number, typename VectorizedArrayType>
class MatrixFree;

template <typename number>
class Vector;
template <typename number>
class FullMatrix;
template <int dim, int spacedim>
class Mapping;
template <typename gridtype>
class InterGridMap;
namespace hp
{
  template <int dim>
  class QCollection;
}
template <typename number>
class AffineConstraints;
#endif

// TODO: Move documentation of functions to the functions!

/**
 * Provide a namespace which offers some operations on vectors. Among these
 * are assembling of standard vectors, integration of the difference of a
 * finite element solution and a continuous function, interpolations and
 * projections of continuous functions to the finite element space and other
 * operations.
 *
 * @note There exist two versions of almost all functions, one that takes an
 * explicit Mapping argument and one that does not. The second one generally
 * calls the first with an implicit $Q_1$ argument (i.e., with an argument of
 * kind MappingQGeneric(1)). If your intend your code to use a different
 * mapping than a (bi-/tri-)linear one, then you need to call the functions
 * <b>with</b> mapping argument should be used.
 *
 *
 * <h3>Description of operations</h3>
 *
 * This collection of methods offers the following operations:
 * <ul>
 * <li> Interpolation: assign each degree of freedom in the vector to be the
 * value of the function given as argument. This is identical to saying that
 * the resulting finite element function (which is isomorphic to the output
 * vector) has exact function values in all support points of trial functions.
 * The support point of a trial function is the point where its value equals
 * one, e.g. for linear trial functions the support points are four corners of
 * an element. This function therefore relies on the assumption that a finite
 * element is used for which the degrees of freedom are function values
 * (Lagrange elements) rather than gradients, normal derivatives, second
 * derivatives, etc (Hermite elements, quintic Argyris element, etc.).
 *
 * It seems inevitable that some values of the vector to be created are set
 * twice or even more than that. The reason is that we have to loop over all
 * cells and get the function values for each of the trial functions located
 * thereon. This applies also to the functions located on faces and corners
 * which we thus visit more than once. While setting the value in the vector
 * is not an expensive operation, the evaluation of the given function may be,
 * taking into account that a virtual function has to be called.
 *
 * <li> Projection: compute the <i>L</i><sup>2</sup>-projection of the given
 * function onto the finite element space, i.e. if <i>f</i> is the function to
 * be projected, compute <i>f<sub>h</sub></i> in <i>V<sub>h</sub></i> such
 * that
 * (<i>f<sub>h</sub></i>,<i>v<sub>h</sub></i>)=(<i>f</i>,<i>v<sub>h</sub></i>)
 * for all discrete test functions <i>v<sub>h</sub></i>. This is done through
 * the solution of the linear system of equations <i> M v = f</i> where
 * <i>M</i> is the mass matrix $m_{ij} = \int_\Omega \phi_i(x) \phi_j(x) dx$
 * and $f_i = \int_\Omega f(x) \phi_i(x) dx$. The solution vector $v$ then is
 * the nodal representation of the projection <i>f<sub>h</sub></i>. The
 * project() functions are used in the step-21 and step-23 tutorial programs.
 *
 * In order to get proper results, it be may necessary to treat boundary
 * conditions right. Below are listed some cases where this may be needed.  If
 * needed, this is done by <i>L</i><sup>2</sup>-projection of the trace of the
 * given function onto the finite element space restricted to the boundary of
 * the domain, then taking this information and using it to eliminate the
 * boundary nodes from the mass matrix of the whole domain, using the
 * MatrixTools::apply_boundary_values() function. The projection of the trace
 * of the function to the boundary is done with the
 * VectorTools::project_boundary_values() (see below) function, which is
 * called with a map of boundary functions
 * std::map<types::boundary_id, const Function<spacedim,number>*> in which all
 * boundary indicators from zero to numbers::internal_face_boundary_id-1
 * (numbers::internal_face_boundary_id is used for other purposes, see the
 * Triangulation class documentation) point to the function to be projected.
 * The projection to the boundary takes place using a second quadrature
 * formula on the boundary given to the project() function. The first
 * quadrature formula is used to compute the right hand side and for numerical
 * quadrature of the mass matrix.
 *
 * The projection of the boundary values first, then eliminating them from the
 * global system of equations is not needed usually. It may be necessary if
 * you want to enforce special restrictions on the boundary values of the
 * projected function, for example in time dependent problems: you may want to
 * project the initial values but need consistency with the boundary values
 * for later times. Since the latter are projected onto the boundary in each
 * time step, it is necessary that we also project the boundary values of the
 * initial values, before projecting them to the whole domain.
 *
 * Obviously, the results of the two schemes for projection are different.
 * Usually, when projecting to the boundary first, the
 * <i>L</i><sup>2</sup>-norm of the difference between original function and
 * projection over the whole domain will be larger (factors of five have been
 * observed) while the <i>L</i><sup>2</sup>-norm of the error integrated over
 * the boundary should of course be less. The reverse should also hold if no
 * projection to the boundary is performed.
 *
 * The selection whether the projection to the boundary first is needed is
 * done with the <tt>project_to_boundary_first</tt> flag passed to the
 * function.  If @p false is given, the additional quadrature formula for
 * faces is ignored.
 *
 * You should be aware of the fact that if no projection to the boundary is
 * requested, a function with zero boundary values may not have zero boundary
 * values after projection. There is a flag for this especially important
 * case, which tells the function to enforce zero boundary values on the
 * respective boundary parts. Since enforced zero boundary values could also
 * have been reached through projection, but are more economically obtain
 * using other methods, the @p project_to_boundary_first flag is ignored if
 * the @p enforce_zero_boundary flag is set.
 *
 * The solution of the linear system is presently done using a simple CG
 * method without preconditioning and without multigrid. This is clearly not
 * too efficient, but sufficient in many cases and simple to implement. This
 * detail may change in the future.
 *
 * <li> Creation of right hand side vectors: The create_right_hand_side()
 * function computes the vector $f_i = \int_\Omega f(x) \phi_i(x) dx$. This is
 * the same as what the <tt>MatrixCreator::create_*</tt> functions which take
 * a right hand side do, but without assembling a matrix.
 *
 * <li> Creation of right hand side vectors for point sources: The
 * create_point_source_vector() function computes the vector $f_i =
 * \int_\Omega \delta(x-x_0) \phi_i(x) dx$.
 *
 * <li> Creation of boundary right hand side vectors: The
 * create_boundary_right_hand_side() function computes the vector $f_i =
 * \int_{\partial\Omega} g(x) \phi_i(x) dx$. This is the right hand side
 * contribution of boundary forces when having inhomogeneous Neumann boundary
 * values in Laplace's equation or other second order operators. This function
 * also takes an optional argument denoting over which parts of the boundary
 * the integration shall extend. If the default argument is used, it is
 * applied to all boundaries.
 *
 * <li> Interpolation of boundary values: The
 * MatrixTools::apply_boundary_values() function takes a list of boundary
 * nodes and their values. You can get such a list by interpolation of a
 * boundary function using the interpolate_boundary_values() function. To use
 * it, you have to specify a list of pairs of boundary indicators (of type
 * <tt>types::boundary_id</tt>; see the section in the documentation of the
 * Triangulation class for more details) and the according functions denoting
 * the Dirichlet boundary values of the nodes on boundary faces with this
 * boundary indicator.
 *
 * Usually, all other boundary conditions, such as inhomogeneous Neumann
 * values or mixed boundary conditions are handled in the weak formulation. No
 * attempt is made to include these into the process of matrix and vector
 * assembly therefore.
 *
 * Within this function, boundary values are interpolated, i.e. a node is
 * given the point value of the boundary function. In some cases, it may be
 * necessary to use the L2-projection of the boundary function or any other
 * method. For this purpose we refer to the project_boundary_values() function
 * below.
 *
 * You should be aware that the boundary function may be evaluated at nodes on
 * the interior of faces. These, however, need not be on the true boundary,
 * but rather are on the approximation of the boundary represented by the
 * mapping of the unit cell to the real cell. Since this mapping will in most
 * cases not be the exact one at the face, the boundary function is evaluated
 * at points which are not on the boundary and you should make sure that the
 * returned values are reasonable in some sense anyway.
 *
 * In 1d the situation is a bit different since there faces (i.e. vertices)
 * have no boundary indicator. It is assumed that if the boundary indicator
 * zero is given in the list of boundary functions, the left boundary point is
 * to be interpolated while the right boundary point is associated with the
 * boundary index 1 in the map. The respective boundary functions are then
 * evaluated at the place of the respective boundary point.
 *
 * <li> Projection of boundary values: The project_boundary_values() function
 * acts similar to the interpolate_boundary_values() function, apart from the
 * fact that it does not get the nodal values of boundary nodes by
 * interpolation but rather through the <i>L</i><sup>2</sup>-projection of the
 * trace of the function to the boundary.
 *
 * The projection takes place on all boundary parts with boundary indicators
 * listed in the map (std::map<types::boundary_id, const
 * Function<spacedim,number>*>) of boundary functions. These boundary parts may
 * or may not be continuous. For these boundary parts, the mass matrix is
 * assembled using the MatrixTools::create_boundary_mass_matrix() function, as
 * well as the appropriate right hand side. Then the resulting system of
 * equations is solved using a simple CG method (without preconditioning), which
 * is in most cases sufficient for the present purpose.
 *
 * <li> Computing errors: The function integrate_difference() performs the
 * calculation of the error between a given (continuous) reference function
 * and the finite element solution in different norms. The integration is
 * performed using a given quadrature formula and assumes that the given
 * finite element objects equals that used for the computation of the
 * solution.
 *
 * The result is stored in a vector (named @p difference), where each entry
 * equals the given norm of the difference on a cell. The order of entries is
 * the same as a @p cell_iterator takes when started with @p begin_active and
 * promoted with the <tt>++</tt> operator.
 *
 * This data, one number per active cell, can be used to generate graphical
 * output by directly passing it to the DataOut class through the
 * DataOut::add_data_vector function. Alternatively, the global error can be
 * computed using VectorTools::compute_global_error(). Finally, the output per
 * cell from VectorTools::integrate_difference() can be interpolated to the
 * nodal points of a finite element field using the
 * DoFTools::distribute_cell_to_dof_vector function.
 *
 * Presently, there is the possibility to compute the following values from
 * the difference, on each cell: @p mean, @p L1_norm, @p L2_norm, @p
 * Linfty_norm, @p H1_seminorm and @p H1_norm, see VectorTools::NormType. For
 * the mean difference value, the reference function minus the numerical
 * solution is computed, not the other way round.
 *
 * The infinity norm of the difference on a given cell returns the maximum
 * absolute value of the difference at the quadrature points given by the
 * quadrature formula parameter. This will in some cases not be too good an
 * approximation, since for example the Gauss quadrature formulae do not
 * evaluate the difference at the end or corner points of the cells. You may
 * want to choose a quadrature formula with more quadrature points or one with
 * another distribution of the quadrature points in this case. You should also
 * take into account the superconvergence properties of finite elements in
 * some points: for example in 1D, the standard finite element method is a
 * collocation method and should return the exact value at nodal points.
 * Therefore, the trapezoidal rule should always return a vanishing L-infinity
 * error. Conversely, in 2D the maximum L-infinity error should be located at
 * the vertices or at the center of the cell, which would make it plausible to
 * use the Simpson quadrature rule. On the other hand, there may be
 * superconvergence at Gauss integration points. These examples are not
 * intended as a rule of thumb, rather they are thought to illustrate that the
 * use of the wrong quadrature formula may show a significantly wrong result
 * and care should be taken to chose the right formula.
 *
 * The <i>H</i><sup>1</sup> seminorm is the <i>L</i><sup>2</sup> norm of the
 * gradient of the difference. The square of the full <i>H</i><sup>1</sup>
 * norm is the sum of the square of seminorm and the square of the
 * <i>L</i><sup>2</sup> norm.
 *
 * To get the global <i>L<sup>1</sup></i> error, you have to sum up the
 * entries in @p difference, e.g. using Vector::l1_norm() function.  For the
 * global <i>L</i><sup>2</sup> difference, you have to sum up the squares of
 * the entries and take the root of the sum, e.g. using Vector::l2_norm().
 * These two operations represent the <i>l</i><sub>1</sub> and
 * <i>l</i><sub>2</sub> norms of the vectors, but you need not take the
 * absolute value of each entry, since the cellwise norms are already
 * positive.
 *
 * To get the global mean difference, simply sum up the elements as above. To
 * get the $L_\infty$ norm, take the maximum of the vector elements, e.g.
 * using the Vector::linfty_norm() function.
 *
 * For the global <i>H</i><sup>1</sup> norm and seminorm, the same rule
 * applies as for the <i>L</i><sup>2</sup> norm: compute the
 * <i>l</i><sub>2</sub> norm of the cell error vector.
 *
 * Note that, in the codimension one case, if you ask for a norm that requires
 * the computation of a gradient, then the provided function is automatically
 * projected along the curve, and the difference is only computed on the
 * tangential part of the gradient, since no information is available on the
 * normal component of the gradient anyway.
 * </ul>
 *
 * All functions use the finite element given to the DoFHandler object the
 * last time that the degrees of freedom were distributed over the
 * triangulation. Also, if access to an object describing the exact form of
 * the boundary is needed, the pointer stored within the triangulation object
 * is accessed.
 *
 * @note Instantiations for this template are provided for some vector types,
 * in particular <code>Vector&lt;float&gt;, Vector&lt;double&gt;,
 * BlockVector&lt;float&gt;, BlockVector&lt;double&gt;</code>; others can be
 * generated in application code (see the section on
 * @ref Instantiations
 * in the manual).
 *
 * @ingroup numerics
 * @author Wolfgang Bangerth, Ralf Hartmann, Guido Kanschat, 1998, 1999, 2000,
 * 2001
 */
namespace VectorTools
{
  /**
   * @name Interpolation and projection
   */
  //@{

  /**
   * Compute the projection of @p function to the finite element space. In other
   * words, given a function $f(\mathbf x)$, the current function computes a
   * finite element function $f_h(\mathbf x)=\sum_j F_j \varphi_j(\mathbf x)$
   * characterized by the (output) vector of nodal values $F$ that satisfies
   * the equation
   * @f{align*}{
   *   (\varphi_i, f_h)_\Omega = (\varphi_i,f)_\Omega
   * @f}
   * for all test functions $\varphi_i$. This requires solving a linear system
   * involving the mass matrix since the equation above is equivalent to
   * the linear system
   * @f{align*}{
   *   \sum_j (\varphi_i, \varphi_j)_\Omega F_j = (\varphi_i,f)_\Omega
   * @f}
   * which can also be written as $MF = \Phi$ with
   * $M_{ij} = (\varphi_i, \varphi_j)_\Omega$ and
   * $\Phi_i = (\varphi_i,f)_\Omega$.
   *
   * By default, no boundary values for $f_h$ are needed nor
   * imposed, but there are optional parameters to this function that allow
   * imposing either zero boundary values or, in a first step, to project
   * the boundary values of $f$ onto the finite element space on the boundary
   * of the mesh in a similar way to above, and then using these values as the
   * imposed boundary values for $f_h$. The ordering of arguments to this
   * function is such that you need not give a second quadrature formula (of
   * type `Quadrature<dim-1>` and used for the computation of the matrix and
   * right hand side for the projection of boundary values) if you
   * don't want to project to the boundary first, but that you must if you want
   * to do so.
   *
   * A MatrixFree implementation is used if the following conditions are met:
   * - @p enforce_zero_boundary is false,
   * - @p project_to_boundary_first is false,
   * - the FiniteElement is supported by the MatrixFree class,
   * - the FiniteElement has less than five components
   * - the degree of the FiniteElement is less than nine.
   * - dim==spacedim
   *
   * In this case, this function performs numerical quadrature using the given
   * quadrature formula for integration of the right hand side $\Phi_i$ while a
   * QGauss(fe_degree+2) object is used for the mass operator. You should
   * therefore make sure that the given quadrature formula is sufficiently
   * accurate for creating the right-hand side.
   *
   * Otherwise, only serial Triangulations are supported and the mass matrix
   * is assembled using MatrixTools::create_mass_matrix. The given
   * quadrature rule is then used for both the matrix and the right-hand side.
   * You should therefore make sure that the given quadrature formula is also
   * sufficient for creating the mass matrix. In particular, the degree of the
   * quadrature formula must be sufficiently high to ensure that the mass
   * matrix is invertible. For example, if you are using a FE_Q(k) element,
   * then the integrand of the matrix entries $M_{ij}$ is of polynomial
   * degree $2k$ in each variable, and you need a Gauss quadrature formula
   * with $k+1$ points in each coordinate direction to ensure that $M$
   * is invertible.
   *
   * See the general documentation of this namespace for further information.
   *
   * In 1d, the default value of the boundary quadrature formula is an invalid
   * object since integration on the boundary doesn't happen in 1d.
   *
   * @param[in] mapping The mapping object to use.
   * @param[in] dof The DoFHandler the describes the finite element space to
   * project into and that corresponds to @p vec.
   * @param[in] constraints Constraints to be used when assembling the mass
   * matrix, typically needed when you have hanging nodes.
   * @param[in] quadrature The quadrature formula to be used for assembling the
   * mass matrix.
   * @param[in] function The function to project into the finite element space.
   * @param[out] vec The output vector where the projected function will be
   * stored in. This vector is required to be already initialized and must not
   * have ghost elements.
   * @param[in] enforce_zero_boundary If true, @p vec will have zero boundary
   * conditions.
   * @param[in] q_boundary Quadrature rule to be used if @p project_to_boundary_first
   * is true.
   * @param[in] project_to_boundary_first If true, perform a projection on the
   * boundary before projecting the interior of the function.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  project(const Mapping<dim, spacedim> &                            mapping,
          const DoFHandler<dim, spacedim> &                         dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const Quadrature<dim> &                                   quadrature,
          const Function<spacedim, typename VectorType::value_type> &function,
          VectorType &                                               vec,
          const bool                 enforce_zero_boundary     = false,
          const Quadrature<dim - 1> &q_boundary                = (dim > 1 ?
                                                     QGauss<dim - 1>(2) :
                                                     Quadrature<dim - 1>(0)),
          const bool                 project_to_boundary_first = false);

  /**
   * Call the project() function above, with
   * <tt>mapping=MappingQGeneric@<dim@>(1)</tt>.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  project(const DoFHandler<dim, spacedim> &                         dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const Quadrature<dim> &                                   quadrature,
          const Function<spacedim, typename VectorType::value_type> &function,
          VectorType &                                               vec,
          const bool                 enforce_zero_boundary     = false,
          const Quadrature<dim - 1> &q_boundary                = (dim > 1 ?
                                                     QGauss<dim - 1>(2) :
                                                     Quadrature<dim - 1>(0)),
          const bool                 project_to_boundary_first = false);

  /**
   * Same as above, but for arguments of type hp::DoFHandler, hp::QCollection,
   * and hp::MappingCollection.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  project(const hp::MappingCollection<dim, spacedim> &              mapping,
          const hp::DoFHandler<dim, spacedim> &                     dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const hp::QCollection<dim> &                              quadrature,
          const Function<spacedim, typename VectorType::value_type> &function,
          VectorType &                                               vec,
          const bool                      enforce_zero_boundary = false,
          const hp::QCollection<dim - 1> &q_boundary = hp::QCollection<dim - 1>(
            dim > 1 ? QGauss<dim - 1>(2) : Quadrature<dim - 1>(0)),
          const bool project_to_boundary_first = false);

  /**
   * Call the project() function above, with a collection of $Q_1$ mapping
   * objects, i.e., with hp::StaticMappingQ1::mapping_collection.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  project(const hp::DoFHandler<dim, spacedim> &                     dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const hp::QCollection<dim> &                              quadrature,
          const Function<spacedim, typename VectorType::value_type> &function,
          VectorType &                                               vec,
          const bool                      enforce_zero_boundary = false,
          const hp::QCollection<dim - 1> &q_boundary = hp::QCollection<dim - 1>(
            dim > 1 ? QGauss<dim - 1>(2) : Quadrature<dim - 1>(0)),
          const bool project_to_boundary_first = false);

  /**
   * The same as above for projection of scalar-valued quadrature data.
   * The user provided function should return a value at the quadrature point
   * based on the cell iterator and quadrature number and of course should be
   * consistent with the provided @p quadrature object, which will be used
   * to assemble the right-hand-side.
   *
   * This function can be used with lambdas:
   * @code
   * VectorTools::project
   * (mapping,
   *  dof_handler,
   *  constraints,
   *  quadrature_formula,
   *  [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
   *       const unsigned int q) -> double
   *  {
   *    return qp_data.get_data(cell)[q]->density;
   *  },
   *  field);
   * @endcode
   * where <code>qp_data</code> is a CellDataStorage object, which stores
   * quadrature point data.
   */
  template <int dim, typename VectorType, int spacedim>
  void
  project(const Mapping<dim, spacedim> &                            mapping,
          const DoFHandler<dim, spacedim> &                         dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const Quadrature<dim> &                                   quadrature,
          const std::function<typename VectorType::value_type(
            const typename DoFHandler<dim, spacedim>::active_cell_iterator &,
            const unsigned int)> &                                  func,
          VectorType &                                              vec_result);

  /**
   * The same as above for projection of scalar-valued MatrixFree quadrature
   * data.
   * The user provided function @p func should return a VectorizedArray value
   * at the quadrature point based on the cell number and quadrature number and
   * should be consistent with the @p n_q_points_1d.
   *
   * This function can be used with lambdas:
   * @code
   * VectorTools::project
   * (matrix_free_data,
   *  constraints,
   *  3,
   *  [&] (const unsigned int cell,
   *       const unsigned int q) -> VectorizedArray<double>
   *  {
   *    return qp_data(cell,q);
   *  },
   *  field);
   * @endcode
   * where <code>qp_data</code> is a an object of type Table<2,
   * VectorizedArray<double> >, which stores quadrature point data.
   *
   * @p fe_component allow to additionally specify which component of @p data
   * to use in case it was constructed with an <code>std::vector<const
   * DoFHandler<dim>*></code>. It will be used internally in constructor of
   * FEEvaluation object.
   */
  template <int dim, typename VectorType>
  void
  project(
    std::shared_ptr<
      const MatrixFree<dim,
                       typename VectorType::value_type,
                       VectorizedArray<typename VectorType::value_type>>> data,
    const AffineConstraints<typename VectorType::value_type> &constraints,
    const unsigned int                                        n_q_points_1d,
    const std::function<VectorizedArray<typename VectorType::value_type>(
      const unsigned int,
      const unsigned int)> &                                  func,
    VectorType &                                              vec_result,
    const unsigned int                                        fe_component = 0);

  /**
   * Same as above but for <code>n_q_points_1d =
   * matrix_free.get_dof_handler().get_fe().degree+1</code>.
   */
  template <int dim, typename VectorType>
  void
  project(
    std::shared_ptr<
      const MatrixFree<dim,
                       typename VectorType::value_type,
                       VectorizedArray<typename VectorType::value_type>>> data,
    const AffineConstraints<typename VectorType::value_type> &constraints,
    const std::function<VectorizedArray<typename VectorType::value_type>(
      const unsigned int,
      const unsigned int)> &                                  func,
    VectorType &                                              vec_result,
    const unsigned int                                        fe_component = 0);


  // @}

} // namespace VectorTools



DEAL_II_NAMESPACE_CLOSE

#endif
