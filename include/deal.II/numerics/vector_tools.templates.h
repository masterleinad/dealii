// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2019 by the deal.II authors
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


#ifndef dealii_vector_tools_templates_h
#define dealii_vector_tools_templates_h

#include <deal.II/base/config.h>

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/polynomials_piecewise.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_epetra_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/range/iterator_range.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <list>
#include <numeric>
#include <set>
#include <typeinfo>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace VectorTools
{
  // This namespace contains the actual implementation called
  // by VectorTools::interpolate and variants (such as
  // VectorTools::interpolate_by_material_id).
  namespace internal
  {
    // A small helper function to transform a component range starting
    // at offset from the real to the unit cell according to the
    // supplied conformity. The function_values vector is transformed
    // in place.
    //
    // FIXME: This should be refactored into the mapping (i.e.
    // implement the inverse function of Mapping<dim, spacedim>::transform).
    // Further, the finite element should make the information about
    // the correct mapping directly accessible (i.e. which MappingKind
    // should be used). Using fe.conforming_space might be a bit of a
    // problem because we only support doing nothing, Hcurl, and Hdiv
    // conforming mappings.
    //
    // Input:
    //  conformity: conformity of the finite element, used to select
    //              appropriate type of transformation
    //  fe_values_jacobians: used for jacobians (and inverses of
    //                        jacobians). the object is supposed to be
    //                        reinit()'d for the current cell
    //  function_values, offset: function_values is manipulated in place
    //                           starting at position offset
    template <int dim, int spacedim, typename FEValuesType, typename T3>
    void
    transform(const typename FiniteElementData<dim>::Conformity conformity,
              const unsigned int                                offset,
              const FEValuesType &fe_values_jacobians,
              T3 &                function_values)
    {
      switch (conformity)
        {
          case FiniteElementData<dim>::Hcurl:
            // See Monk, Finite Element Methods for Maxwell's Equations,
            // p. 77ff, formula (3.76) and Corollary 3.58.
            // For given mapping F_K: \hat K \to K, we have to transform
            //  \hat u = (dF_K)^T u\circ F_K

            for (unsigned int i = 0; i < function_values.size(); ++i)
              {
                const auto &jacobians =
                  fe_values_jacobians.get_present_fe_values().get_jacobians();

                const ArrayView<typename T3::value_type::value_type> source(
                  &function_values[i][0] + offset, dim);

                Tensor<1,
                       dim,
                       typename ProductType<typename T3::value_type::value_type,
                                            double>::type>
                  destination;

                // value[m] <- sum jacobian_transpose[m][n] * old_value[n]:
                TensorAccessors::contract<1, 2, 1, dim>(
                  destination, jacobians[i].transpose(), source);

                // now copy things back into the input=output vector
                for (unsigned int d = 0; d < dim; ++d)
                  source[d] = destination[d];
              }
            break;

          case FiniteElementData<dim>::Hdiv:
            // See Monk, Finite Element Methods for Maxwell's Equations,
            // p. 79ff, formula (3.77) and Lemma 3.59.
            // For given mapping F_K: \hat K \to K, we have to transform
            //  \hat w = det(dF_K) (dF_K)^{-1} w\circ F_K

            for (unsigned int i = 0; i < function_values.size(); ++i)
              {
                const auto &jacobians =
                  fe_values_jacobians.get_present_fe_values().get_jacobians();
                const auto &inverse_jacobians =
                  fe_values_jacobians.get_present_fe_values()
                    .get_inverse_jacobians();

                const ArrayView<typename T3::value_type::value_type> source(
                  &function_values[i][0] + offset, dim);

                Tensor<1,
                       dim,
                       typename ProductType<typename T3::value_type::value_type,
                                            double>::type>
                  destination;

                // value[m] <- sum inverse_jacobians[m][n] * old_value[n]:
                TensorAccessors::contract<1, 2, 1, dim>(destination,
                                                        inverse_jacobians[i],
                                                        source);
                destination *= jacobians[i].determinant();

                // now copy things back into the input=output vector
                for (unsigned int d = 0; d < dim; ++d)
                  source[d] = destination[d];
              }
            break;

          case FiniteElementData<dim>::H1:
            DEAL_II_FALLTHROUGH;
          case FiniteElementData<dim>::L2:
            // See Monk, Finite Element Methods for Maxwell's Equations,
            // p. 77ff, formula (3.74).
            // For given mapping F_K: \hat K \to K, we have to transform
            //  \hat p = p\circ F_K
            //  i.e., do nothing.
            break;

          default:
            // In case we deal with an unknown conformity, just assume we
            // deal with a Lagrange element and do nothing.
            break;

        } /*switch*/
    }


    // A small helper function that iteratively applies above transform
    // function to a vector function_values recursing over a given finite
    // element decomposing it into base elements:
    //
    // Input
    //   fe: the full finite element corresponding to function_values
    //   [ rest see above]
    // Output: the offset after we have handled the element at
    //   a given offset
    template <int dim, int spacedim, typename FEValuesType, typename T3>
    unsigned int
    apply_transform(const FiniteElement<dim, spacedim> &fe,
                    const unsigned int                  offset,
                    const FEValuesType &                fe_values_jacobians,
                    T3 &                                function_values)
    {
      if (const auto *system =
            dynamic_cast<const FESystem<dim, spacedim> *>(&fe))
        {
          // In case of an FESystem transform every (vector) component
          // separately:
          unsigned current_offset = offset;
          for (unsigned int i = 0; i < system->n_base_elements(); ++i)
            {
              const auto &base_fe      = system->base_element(i);
              const auto  multiplicity = system->element_multiplicity(i);
              for (unsigned int m = 0; m < multiplicity; ++m)
                {
                  // recursively call apply_transform to make sure to
                  // correctly handle nested fe systems.
                  current_offset = apply_transform(base_fe,
                                                   current_offset,
                                                   fe_values_jacobians,
                                                   function_values);
                }
            }
          return current_offset;
        }
      else
        {
          transform<dim, spacedim>(fe.conforming_space,
                                   offset,
                                   fe_values_jacobians,
                                   function_values);
          return (offset + fe.n_components());
        }
    }


    // Internal implementation of interpolate that takes a generic functor
    // function such that function(cell) is of type
    // Function<spacedim, typename VectorType::value_type>*
    //
    // A given cell is skipped if function(cell) == nullptr
    template <int dim,
              int spacedim,
              typename VectorType,
              template <int, int> class DoFHandlerType,
              typename T>
    void
    interpolate(const Mapping<dim, spacedim> &       mapping,
                const DoFHandlerType<dim, spacedim> &dof_handler,
                T &                                  function,
                VectorType &                         vec,
                const ComponentMask &                component_mask)
    {
      Assert(component_mask.represents_n_components(
               dof_handler.get_fe_collection().n_components()),
             ExcMessage(
               "The number of components in the mask has to be either "
               "zero or equal to the number of components in the finite "
               "element."));

      Assert(vec.size() == dof_handler.n_dofs(),
             ExcDimensionMismatch(vec.size(), dof_handler.n_dofs()));

      Assert(component_mask.n_selected_components(
               dof_handler.get_fe_collection().n_components()) > 0,
             ComponentMask::ExcNoComponentSelected());

      //
      // Computing the generalized interpolant isn't quite as straightforward
      // as for classical Lagrange elements. A major complication is the fact
      // it generally doesn't hold true that a function evaluates to the same
      // dof coefficient on different cells. This means *setting* the value
      // of a (global) degree of freedom computed on one cell doesn't
      // necessarily lead to the same result when computed on a neighboring
      // cell (that shares the same global degree of freedom).
      //
      // We thus, do the following operation:
      //
      // On each cell:
      //
      //  - We first determine all function values u(x_i) in generalized
      //    support points
      //
      //  - We transform these function values back to the unit cell
      //    according to the conformity of the component (scalar, Hdiv, or
      //    Hcurl conforming); see [Monk, Finite Element Methods for Maxwell's
      //    Equations, p.77ff Section 3.9] for details. This results in
      //    \hat u(\hat x_i)
      //
      //  - We convert these generalized support point values to nodal values
      //
      //  - For every global dof we take the average 1 / n_K \sum_{K} dof_K
      //    where n_K is the number of cells sharing the global dof and dof_K
      //    is the computed value on the cell K.
      //
      // For every degree of freedom that is shared by k cells, we compute
      // its value on all k cells and take the weighted average with respect
      // to the JxW values.
      //

      using number = typename VectorType::value_type;

      const hp::FECollection<dim, spacedim> &fe(
        dof_handler.get_fe_collection());

      std::vector<types::global_dof_index> dofs_on_cell(fe.max_dofs_per_cell());

      // Temporary storage for cell-wise interpolation operation. We store a
      // variant for every fe we encounter to speed up resizing operations.
      // The first vector is used for local function evaluation. The vector
      // dof_values is used to store intermediate cell-wise interpolation
      // results (see the detailed explanation in the for loop further down
      // below).

      std::vector<std::vector<Vector<number>>> fe_function_values(fe.size());
      std::vector<std::vector<number>>         fe_dof_values(fe.size());

      // We will need two temporary global vectors that store the new values
      // and weights.
      VectorType interpolation;
      VectorType weights;
      interpolation.reinit(vec);
      weights.reinit(vec);

      // Store locally owned dofs, so that we can skip all non-local dofs,
      // if they do not need to be interpolated.
      const IndexSet locally_owned_dofs = vec.locally_owned_elements();

      // We use an FEValues object to transform all generalized support
      // points from the unit cell to the real cell coordinates. Thus,
      // initialize a quadrature with all generalized support points and
      // create an FEValues object with it.

      hp::QCollection<dim> support_quadrature;
      for (unsigned int fe_index = 0; fe_index < fe.size(); ++fe_index)
        {
          const auto &points = fe[fe_index].get_generalized_support_points();
          support_quadrature.push_back(Quadrature<dim>(points));
        }

      const hp::MappingCollection<dim, spacedim> mapping_collection(mapping);

      // An FEValues object to evaluate (generalized) support point
      // locations as well as Jacobians and their inverses.
      // the latter are only needed for Hcurl or Hdiv conforming elements,
      // but we'll just always include them.
      hp::FEValues<dim, spacedim> fe_values(mapping_collection,
                                            fe,
                                            support_quadrature,
                                            update_quadrature_points |
                                              update_jacobians |
                                              update_inverse_jacobians);

      //
      // Now loop over all locally owned, active cells.
      //

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          // If this cell is not locally owned, do nothing.
          if (!cell->is_locally_owned())
            continue;

          const unsigned int fe_index = cell->active_fe_index();

          // Do nothing if there are no local degrees of freedom.
          if (fe[fe_index].dofs_per_cell == 0)
            continue;

          // Skip processing of the current cell if the function object is
          // invalid. This is used by interpolate_by_material_id to skip
          // interpolating over cells with unknown material id.
          if (!function(cell))
            continue;

          // Get transformed, generalized support points
          fe_values.reinit(cell);
          const std::vector<Point<spacedim>> &generalized_support_points =
            fe_values.get_present_fe_values().get_quadrature_points();

          // Get indices of the dofs on this cell
          const auto n_dofs = fe[fe_index].dofs_per_cell;
          dofs_on_cell.resize(n_dofs);
          cell->get_dof_indices(dofs_on_cell);

          // Prepare temporary storage
          auto &function_values = fe_function_values[fe_index];
          auto &dof_values      = fe_dof_values[fe_index];

          const auto n_components = fe[fe_index].n_components();
          function_values.resize(generalized_support_points.size(),
                                 Vector<number>(n_components));
          dof_values.resize(n_dofs);

          // Get all function values:
          Assert(
            n_components == function(cell)->n_components,
            ExcDimensionMismatch(dof_handler.get_fe_collection().n_components(),
                                 function(cell)->n_components));
          function(cell)->vector_value_list(generalized_support_points,
                                            function_values);

          {
            // Before we can average, we have to transform all function values
            // from the real cell back to the unit cell. We query the finite
            // element for the correct transformation. Matters get a bit more
            // complicated because we have to apply said transformation for
            // every base element.

            const unsigned int offset =
              apply_transform(fe[fe_index],
                              /* starting_offset = */ 0,
                              fe_values,
                              function_values);
            (void)offset;
            Assert(offset == n_components, ExcInternalError());
          }

          FETools::convert_generalized_support_point_values_to_dof_values(
            fe[fe_index], function_values, dof_values);

          for (unsigned int i = 0; i < n_dofs; ++i)
            {
              const auto &nonzero_components =
                fe[fe_index].get_nonzero_components(i);

              // Figure out whether the component mask applies. We assume
              // that we are allowed to set degrees of freedom if at least
              // one of the components (of the dof) is selected.
              bool selected = false;
              for (unsigned int c = 0; c < nonzero_components.size(); ++c)
                selected =
                  selected || (nonzero_components[c] && component_mask[c]);

              if (selected)
                {
#ifdef DEBUG
                  // make sure that all selected base elements are indeed
                  // interpolatory

                  if (const auto fe_system =
                        dynamic_cast<const FESystem<dim> *>(&fe[fe_index]))
                    {
                      const auto index =
                        fe_system->system_to_base_index(i).first.first;
                      Assert(fe_system->base_element(index)
                               .has_generalized_support_points(),
                             ExcMessage("The component mask supplied to "
                                        "VectorTools::interpolate selects a "
                                        "non-interpolatory element."));
                    }
#endif

                  // Add local values to the global vectors
                  ::dealii::internal::ElementAccess<VectorType>::add(
                    dof_values[i], dofs_on_cell[i], interpolation);
                  ::dealii::internal::ElementAccess<VectorType>::add(
                    typename VectorType::value_type(1.0),
                    dofs_on_cell[i],
                    weights);
                }
              else
                {
                  // If a component is ignored, copy the dof values
                  // from the vector "vec", but only if they are locally
                  // available
                  if (locally_owned_dofs.is_element(dofs_on_cell[i]))
                    {
                      const auto value =
                        ::dealii::internal::ElementAccess<VectorType>::get(
                          vec, dofs_on_cell[i]);
                      ::dealii::internal::ElementAccess<VectorType>::add(
                        value, dofs_on_cell[i], interpolation);
                      ::dealii::internal::ElementAccess<VectorType>::add(
                        typename VectorType::value_type(1.0),
                        dofs_on_cell[i],
                        weights);
                    }
                }
            }
        } /* loop over dof_handler.active_cell_iterators() */

      interpolation.compress(VectorOperation::add);
      weights.compress(VectorOperation::add);

      for (const auto i : interpolation.locally_owned_elements())
        {
          const auto weight =
            ::dealii::internal::ElementAccess<VectorType>::get(weights, i);

          // See if we touched this DoF at all. If so, set the average
          // of the value we computed in the output vector. Otherwise,
          // don't touch the value at all.
          if (weight != number(0))
            {
              const auto value =
                ::dealii::internal::ElementAccess<VectorType>::get(
                  interpolation, i);
              ::dealii::internal::ElementAccess<VectorType>::set(value / weight,
                                                                 i,
                                                                 vec);
            }
        }
      vec.compress(VectorOperation::insert);
    }

  } // namespace internal



  template <int dim,
            int spacedim,
            typename VectorType,
            template <int, int> class DoFHandlerType>
  void
  interpolate(
    const Mapping<dim, spacedim> &                             mapping,
    const DoFHandlerType<dim, spacedim> &                      dof_handler,
    const Function<spacedim, typename VectorType::value_type> &function,
    VectorType &                                               vec,
    const ComponentMask &                                      component_mask)
  {
    Assert(dof_handler.get_fe_collection().n_components() ==
             function.n_components,
           ExcDimensionMismatch(dof_handler.get_fe_collection().n_components(),
                                function.n_components));

    // Create a small lambda capture wrapping function and call the
    // internal implementation
    const auto function_map = [&function](
      const typename DoFHandlerType<dim, spacedim>::active_cell_iterator &)
      -> const Function<spacedim, typename VectorType::value_type> *
    {
      return &function;
    };

    internal::interpolate(
      mapping, dof_handler, function_map, vec, component_mask);
  }



  template <int dim,
            int spacedim,
            typename VectorType,
            template <int, int> class DoFHandlerType>
  void
  interpolate(
    const DoFHandlerType<dim, spacedim> &                      dof,
    const Function<spacedim, typename VectorType::value_type> &function,
    VectorType &                                               vec,
    const ComponentMask &                                      component_mask)
  {
    interpolate(StaticMappingQ1<dim, spacedim>::mapping,
                dof,
                function,
                vec,
                component_mask);
  }



  template <int dim, class InVector, class OutVector, int spacedim>
  void
  interpolate(const DoFHandler<dim, spacedim> &dof_1,
              const DoFHandler<dim, spacedim> &dof_2,
              const FullMatrix<double> &       transfer,
              const InVector &                 data_1,
              OutVector &                      data_2)
  {
    using number = typename OutVector::value_type;
    Vector<number> cell_data_1(dof_1.get_fe().dofs_per_cell);
    Vector<number> cell_data_2(dof_2.get_fe().dofs_per_cell);

    // Reset output vector.
    data_2 = static_cast<number>(0);

    // Store how many cells share each dof (unghosted).
    OutVector touch_count;
    touch_count.reinit(data_2);

    std::vector<types::global_dof_index> local_dof_indices(
      dof_2.get_fe().dofs_per_cell);

    typename DoFHandler<dim, spacedim>::active_cell_iterator cell_1 =
      dof_1.begin_active();
    typename DoFHandler<dim, spacedim>::active_cell_iterator cell_2 =
      dof_2.begin_active();
    const typename DoFHandler<dim, spacedim>::cell_iterator end_1 = dof_1.end();

    for (; cell_1 != end_1; ++cell_1, ++cell_2)
      {
        if (cell_1->is_locally_owned())
          {
            Assert(cell_2->is_locally_owned(), ExcInternalError());

            // Perform dof interpolation.
            cell_1->get_dof_values(data_1, cell_data_1);
            transfer.vmult(cell_data_2, cell_data_1);

            cell_2->get_dof_indices(local_dof_indices);

            // Distribute cell vector.
            for (unsigned int j = 0; j < dof_2.get_fe().dofs_per_cell; ++j)
              {
                ::dealii::internal::ElementAccess<OutVector>::add(
                  cell_data_2(j), local_dof_indices[j], data_2);

                // Count cells that share each dof.
                ::dealii::internal::ElementAccess<OutVector>::add(
                  static_cast<number>(1), local_dof_indices[j], touch_count);
              }
          }
      }

    // Collect information over all the parallel processes.
    data_2.compress(VectorOperation::add);
    touch_count.compress(VectorOperation::add);

    // Compute the mean value of the sum which has been placed in
    // each entry of the output vector only at locally owned elements.
    for (const auto &i : data_2.locally_owned_elements())
      {
        const number touch_count_i =
          ::dealii::internal::ElementAccess<OutVector>::get(touch_count, i);

        Assert(touch_count_i != static_cast<number>(0), ExcInternalError());

        const number value =
          ::dealii::internal::ElementAccess<OutVector>::get(data_2, i) /
          touch_count_i;

        ::dealii::internal::ElementAccess<OutVector>::set(value, i, data_2);
      }

    // Compress data_2 to set the proper values on all the parallel processes.
    data_2.compress(VectorOperation::insert);
  }



  template <int dim,
            int spacedim,
            typename VectorType,
            template <int, int> class DoFHandlerType>
  void
  interpolate_based_on_material_id(
    const Mapping<dim, spacedim> &       mapping,
    const DoFHandlerType<dim, spacedim> &dof_handler,
    const std::map<types::material_id,
                   const Function<spacedim, typename VectorType::value_type> *>
      &                  functions,
    VectorType &         vec,
    const ComponentMask &component_mask)
  {
    // Create a small lambda capture wrapping the function map and call the
    // internal implementation
    const auto function_map = [&functions](
      const typename DoFHandlerType<dim, spacedim>::active_cell_iterator &cell)
      -> const Function<spacedim, typename VectorType::value_type> *
    {
      const auto function = functions.find(cell->material_id());
      if (function != functions.end())
        return function->second;
      else
        return nullptr;
    };

    internal::interpolate(
      mapping, dof_handler, function_map, vec, component_mask);
  }


  namespace internal
  {
    /**
     * Interpolate zero boundary values. We don't need to worry about a
     * mapping here because the function we evaluate for the DoFs is zero in
     * the mapped locations as well as in the original, unmapped locations
     */
    template <int dim,
              int spacedim,
              template <int, int> class DoFHandlerType,
              typename number>
    void
    interpolate_zero_boundary_values(
      const DoFHandlerType<dim, spacedim> &      dof_handler,
      std::map<types::global_dof_index, number> &boundary_values)
    {
      // loop over all boundary faces
      // to get all dof indices of
      // dofs on the boundary. note
      // that in 3d there are cases
      // where a face is not at the
      // boundary, yet one of its
      // lines is, and we should
      // consider the degrees of
      // freedom on it as boundary
      // nodes. likewise, in 2d and
      // 3d there are cases where a
      // cell is only at the boundary
      // by one vertex. nevertheless,
      // since we do not support
      // boundaries with dimension
      // less or equal to dim-2, each
      // such boundary dof is also
      // found from some other face
      // that is actually wholly on
      // the boundary, not only by
      // one line or one vertex
      typename DoFHandlerType<dim, spacedim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
      std::vector<types::global_dof_index> face_dof_indices;
      for (; cell != endc; ++cell)
        for (auto f : GeometryInfo<dim>::face_indices())
          if (cell->at_boundary(f))
            {
              face_dof_indices.resize(cell->get_fe().dofs_per_face);
              cell->face(f)->get_dof_indices(face_dof_indices,
                                             cell->active_fe_index());
              for (unsigned int i = 0; i < cell->get_fe().dofs_per_face; ++i)
                // enter zero boundary values
                // for all boundary nodes
                //
                // we need not care about
                // vector valued elements here,
                // since we set all components
                boundary_values[face_dof_indices[i]] = 0.;
            }
    }
  } // namespace internal



  template <int dim,
            int spacedim,
            typename VectorType,
            template <int, int> class DoFHandlerType>
  void
  interpolate_to_different_mesh(const DoFHandlerType<dim, spacedim> &dof1,
                                const VectorType &                   u1,
                                const DoFHandlerType<dim, spacedim> &dof2,
                                VectorType &                         u2)
  {
    Assert(GridTools::have_same_coarse_mesh(dof1, dof2),
           ExcMessage("The two DoF handlers must represent triangulations that "
                      "have the same coarse meshes"));

    InterGridMap<DoFHandlerType<dim, spacedim>> intergridmap;
    intergridmap.make_mapping(dof1, dof2);

    AffineConstraints<typename VectorType::value_type> dummy;
    dummy.close();

    interpolate_to_different_mesh(intergridmap, u1, dummy, u2);
  }



  template <int dim,
            int spacedim,
            typename VectorType,
            template <int, int> class DoFHandlerType>
  void
  interpolate_to_different_mesh(
    const DoFHandlerType<dim, spacedim> &                     dof1,
    const VectorType &                                        u1,
    const DoFHandlerType<dim, spacedim> &                     dof2,
    const AffineConstraints<typename VectorType::value_type> &constraints,
    VectorType &                                              u2)
  {
    Assert(GridTools::have_same_coarse_mesh(dof1, dof2),
           ExcMessage("The two DoF handlers must represent triangulations that "
                      "have the same coarse meshes"));

    InterGridMap<DoFHandlerType<dim, spacedim>> intergridmap;
    intergridmap.make_mapping(dof1, dof2);

    interpolate_to_different_mesh(intergridmap, u1, constraints, u2);
  }

  namespace internal
  {
    /**
     * Return whether the cell and all of its descendants are locally owned.
     */
    template <typename cell_iterator>
    bool
    is_locally_owned(const cell_iterator &cell)
    {
      if (cell->is_active())
        return cell->is_locally_owned();

      for (unsigned int c = 0; c < cell->n_children(); ++c)
        if (!is_locally_owned(cell->child(c)))
          return false;

      return true;
    }
  } // namespace internal

  template <int dim,
            int spacedim,
            typename VectorType,
            template <int, int> class DoFHandlerType>
  void
  interpolate_to_different_mesh(
    const InterGridMap<DoFHandlerType<dim, spacedim>> &       intergridmap,
    const VectorType &                                        u1,
    const AffineConstraints<typename VectorType::value_type> &constraints,
    VectorType &                                              u2)
  {
    const DoFHandlerType<dim, spacedim> &dof1 = intergridmap.get_source_grid();
    const DoFHandlerType<dim, spacedim> &dof2 =
      intergridmap.get_destination_grid();
    (void)dof2;

    Assert(dof1.get_fe_collection() == dof2.get_fe_collection(),
           ExcMessage(
             "The FECollections of both DoFHandler objects must match"));
    Assert(u1.size() == dof1.n_dofs(),
           ExcDimensionMismatch(u1.size(), dof1.n_dofs()));
    Assert(u2.size() == dof2.n_dofs(),
           ExcDimensionMismatch(u2.size(), dof2.n_dofs()));

    Vector<typename VectorType::value_type> cache;

    // Looping over the finest common
    // mesh, this means that source and
    // destination cells have to be on the
    // same level and at least one has to
    // be active.
    //
    // Therefore, loop over all cells
    // (active and inactive) of the source
    // grid ..
    typename DoFHandlerType<dim, spacedim>::cell_iterator cell1 = dof1.begin();
    const typename DoFHandlerType<dim, spacedim>::cell_iterator endc1 =
      dof1.end();

    for (; cell1 != endc1; ++cell1)
      {
        const typename DoFHandlerType<dim, spacedim>::cell_iterator cell2 =
          intergridmap[cell1];

        // .. and skip if source and destination
        // cells are not on the same level ..
        if (cell1->level() != cell2->level())
          continue;
        // .. or none of them is active.
        if (!cell1->is_active() && !cell2->is_active())
          continue;

        Assert(
          internal::is_locally_owned(cell1) ==
            internal::is_locally_owned(cell2),
          ExcMessage(
            "The two Triangulations are required to have the same parallel partitioning."));

        // Skip foreign cells.
        if (cell1->is_active() && !cell1->is_locally_owned())
          continue;
        if (cell2->is_active() && !cell2->is_locally_owned())
          continue;

        // Get and set the corresponding
        // dof_values by interpolation.
        if (cell1->is_active())
          {
            cache.reinit(cell1->get_fe().dofs_per_cell);
            cell1->get_interpolated_dof_values(u1,
                                               cache,
                                               cell1->active_fe_index());
            cell2->set_dof_values_by_interpolation(cache,
                                                   u2,
                                                   cell1->active_fe_index());
          }
        else
          {
            cache.reinit(cell2->get_fe().dofs_per_cell);
            cell1->get_interpolated_dof_values(u1,
                                               cache,
                                               cell2->active_fe_index());
            cell2->set_dof_values_by_interpolation(cache,
                                                   u2,
                                                   cell2->active_fe_index());
          }
      }

    // finish the work on parallel vectors
    u2.compress(VectorOperation::insert);
    // Apply hanging node constraints.
    constraints.distribute(u2);
  }

  namespace internal
  {
    /**
     * Compute the boundary values to be used in the project() functions.
     */
    template <int dim,
              int spacedim,
              template <int, int> class DoFHandlerType,
              template <int, int> class M_or_MC,
              template <int> class Q_or_QC,
              typename number>
    void
    project_compute_b_v(
      const M_or_MC<dim, spacedim> &             mapping,
      const DoFHandlerType<dim, spacedim> &      dof,
      const Function<spacedim, number> &         function,
      const bool                                 enforce_zero_boundary,
      const Q_or_QC<dim - 1> &                   q_boundary,
      const bool                                 project_to_boundary_first,
      std::map<types::global_dof_index, number> &boundary_values)
    {
      if (enforce_zero_boundary == true)
        // no need to project boundary
        // values, but enforce
        // homogeneous boundary values
        // anyway
        interpolate_zero_boundary_values(dof, boundary_values);

      else
        // no homogeneous boundary values
        if (project_to_boundary_first == true)
        // boundary projection required
        {
          // set up a list of boundary
          // functions for the
          // different boundary
          // parts. We want the
          // function to hold on
          // all parts of the boundary
          const std::vector<types::boundary_id> used_boundary_ids =
            dof.get_triangulation().get_boundary_ids();

          std::map<types::boundary_id, const Function<spacedim, number> *>
            boundary_functions;
          for (const auto used_boundary_id : used_boundary_ids)
            boundary_functions[used_boundary_id] = &function;
          project_boundary_values(
            mapping, dof, boundary_functions, q_boundary, boundary_values);
        }
    }



    /**
     * Return whether the boundary values try to constrain a degree of freedom
     * that is already constrained to something else
     */
    template <typename number>
    bool
    constraints_and_b_v_are_compatible(
      const AffineConstraints<number> &          constraints,
      std::map<types::global_dof_index, number> &boundary_values)
    {
      for (const auto &boundary_value : boundary_values)
        if (constraints.is_constrained(boundary_value.first))
          // TODO: This looks wrong -- shouldn't it be ==0 in the first
          // condition and && ?
          if (!(constraints.get_constraint_entries(boundary_value.first)
                    ->size() > 0 ||
                (constraints.get_inhomogeneity(boundary_value.first) ==
                 boundary_value.second)))
            return false;

      return true;
    }



    /**
     * Generic implementation of the project() function
     */
    template <int dim,
              int spacedim,
              typename VectorType,
              template <int, int> class DoFHandlerType,
              template <int, int> class M_or_MC,
              template <int> class Q_or_QC>
    void
    do_project(
      const M_or_MC<dim, spacedim> &                             mapping,
      const DoFHandlerType<dim, spacedim> &                      dof,
      const AffineConstraints<typename VectorType::value_type> & constraints,
      const Q_or_QC<dim> &                                       quadrature,
      const Function<spacedim, typename VectorType::value_type> &function,
      VectorType &                                               vec_result,
      const bool              enforce_zero_boundary,
      const Q_or_QC<dim - 1> &q_boundary,
      const bool              project_to_boundary_first)
    {
      using number = typename VectorType::value_type;
      Assert(dof.get_fe(0).n_components() == function.n_components,
             ExcDimensionMismatch(dof.get_fe(0).n_components(),
                                  function.n_components));
      Assert(vec_result.size() == dof.n_dofs(),
             ExcDimensionMismatch(vec_result.size(), dof.n_dofs()));

      // make up boundary values
      std::map<types::global_dof_index, number> boundary_values;
      project_compute_b_v(mapping,
                          dof,
                          function,
                          enforce_zero_boundary,
                          q_boundary,
                          project_to_boundary_first,
                          boundary_values);

      // check if constraints are compatible (see below)
      const bool constraints_are_compatible =
        constraints_and_b_v_are_compatible<number>(constraints,
                                                   boundary_values);

      // set up mass matrix and right hand side
      Vector<number>  vec(dof.n_dofs());
      SparsityPattern sparsity;
      {
        DynamicSparsityPattern dsp(dof.n_dofs(), dof.n_dofs());
        DoFTools::make_sparsity_pattern(dof,
                                        dsp,
                                        constraints,
                                        !constraints_are_compatible);

        sparsity.copy_from(dsp);
      }
      SparseMatrix<number> mass_matrix(sparsity);
      Vector<number>       tmp(mass_matrix.n());

      // If the constraints object does not conflict with the given boundary
      // values (i.e., it either does not contain boundary values or it contains
      // the same as boundary_values), we can let it call
      // distribute_local_to_global straight away, otherwise we need to first
      // interpolate the boundary values and then condense the matrix and vector
      if (constraints_are_compatible)
        {
          const Function<spacedim, number> *dummy = nullptr;
          MatrixCreator::create_mass_matrix(mapping,
                                            dof,
                                            quadrature,
                                            mass_matrix,
                                            function,
                                            tmp,
                                            dummy,
                                            constraints);
          if (boundary_values.size() > 0)
            MatrixTools::apply_boundary_values(
              boundary_values, mass_matrix, vec, tmp, true);
        }
      else
        {
          // create mass matrix and rhs at once, which is faster.
          MatrixCreator::create_mass_matrix(
            mapping, dof, quadrature, mass_matrix, function, tmp);
          MatrixTools::apply_boundary_values(
            boundary_values, mass_matrix, vec, tmp, true);
          constraints.condense(mass_matrix, tmp);
        }

      invert_mass_matrix(mass_matrix, tmp, vec);
      constraints.distribute(vec);

      // copy vec into vec_result. we can't use vec_result itself above, since
      // it may be of another type than Vector<double> and that wouldn't
      // necessarily go together with the matrix and other functions
      for (unsigned int i = 0; i < vec.size(); ++i)
        ::dealii::internal::ElementAccess<VectorType>::set(vec(i),
                                                           i,
                                                           vec_result);
    }



    /*
     * MatrixFree implementation of project() for an arbitrary number of
     * components and arbitrary degree of the FiniteElement.
     */
    template <int components,
              int fe_degree,
              int dim,
              typename Number,
              int spacedim>
    void
    project_matrix_free(
      const Mapping<dim, spacedim> &   mapping,
      const DoFHandler<dim, spacedim> &dof,
      const AffineConstraints<Number> &constraints,
      const Quadrature<dim> &          quadrature,
      const Function<
        spacedim,
        typename LinearAlgebra::distributed::Vector<Number>::value_type>
        &                                         function,
      LinearAlgebra::distributed::Vector<Number> &work_result,
      const bool                                  enforce_zero_boundary,
      const Quadrature<dim - 1> &                 q_boundary,
      const bool                                  project_to_boundary_first)
    {
      Assert(project_to_boundary_first == false, ExcNotImplemented());
      Assert(enforce_zero_boundary == false, ExcNotImplemented());
      (void)enforce_zero_boundary;
      (void)project_to_boundary_first;
      (void)q_boundary;

      Assert(dof.get_fe(0).n_components() == function.n_components,
             ExcDimensionMismatch(dof.get_fe(0).n_components(),
                                  function.n_components));
      Assert(fe_degree == -1 ||
               dof.get_fe().degree == static_cast<unsigned int>(fe_degree),
             ExcDimensionMismatch(fe_degree, dof.get_fe().degree));
      Assert(dof.get_fe(0).n_components() == components,
             ExcDimensionMismatch(components, dof.get_fe(0).n_components()));

      // set up mass matrix and right hand side
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::partition_color;
      additional_data.mapping_update_flags =
        (update_values | update_JxW_values);
      std::shared_ptr<MatrixFree<dim, Number>> matrix_free(
        new MatrixFree<dim, Number>());
      matrix_free->reinit(mapping,
                          dof,
                          constraints,
                          QGauss<1>(dof.get_fe().degree + 2),
                          additional_data);
      using MatrixType = MatrixFreeOperators::MassOperator<
        dim,
        fe_degree,
        fe_degree + 2,
        components,
        LinearAlgebra::distributed::Vector<Number>>;
      MatrixType mass_matrix;
      mass_matrix.initialize(matrix_free);
      mass_matrix.compute_diagonal();

      LinearAlgebra::distributed::Vector<Number> rhs, inhomogeneities;
      matrix_free->initialize_dof_vector(work_result);
      matrix_free->initialize_dof_vector(rhs);
      matrix_free->initialize_dof_vector(inhomogeneities);
      constraints.distribute(inhomogeneities);
      inhomogeneities *= -1.;

      {
        create_right_hand_side(
          mapping, dof, quadrature, function, rhs, constraints);

        // account for inhomogeneous constraints
        inhomogeneities.update_ghost_values();
        FEEvaluation<dim, fe_degree, fe_degree + 2, components, Number> phi(
          *matrix_free);
        for (unsigned int cell = 0; cell < matrix_free->n_macro_cells(); ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values_plain(inhomogeneities);
            phi.evaluate(true, false);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_value(phi.get_value(q), q);

            phi.integrate(true, false);
            phi.distribute_local_to_global(rhs);
          }
        rhs.compress(VectorOperation::add);
      }

      // now invert the matrix
      // Allow for a maximum of 6*n steps to reduce the residual by 10^-12. n
      // steps may not be sufficient, since roundoff errors may accumulate for
      // badly conditioned matrices. This behavior can be observed, e.g. for
      // FE_Q_Hierarchical for degree higher than three.
      ReductionControl control(6 * rhs.size(), 0., 1e-12, false, false);
      SolverCG<LinearAlgebra::distributed::Vector<Number>> cg(control);
      PreconditionJacobi<MatrixType>                       preconditioner;
      preconditioner.initialize(mass_matrix, 1.);
      cg.solve(mass_matrix, work_result, rhs, preconditioner);
      work_result += inhomogeneities;

      constraints.distribute(work_result);
    }



    /**
     * Helper interface. After figuring out the number of components in
     * project_matrix_free_component, we determine the degree of the
     * FiniteElement and call project_matrix_free with the appropriate
     * template arguments.
     */
    template <int components, int dim, typename Number, int spacedim>
    void
    project_matrix_free_degree(
      const Mapping<dim, spacedim> &   mapping,
      const DoFHandler<dim, spacedim> &dof,
      const AffineConstraints<Number> &constraints,
      const Quadrature<dim> &          quadrature,
      const Function<
        spacedim,
        typename LinearAlgebra::distributed::Vector<Number>::value_type>
        &                                         function,
      LinearAlgebra::distributed::Vector<Number> &work_result,
      const bool                                  enforce_zero_boundary,
      const Quadrature<dim - 1> &                 q_boundary,
      const bool                                  project_to_boundary_first)
    {
      switch (dof.get_fe().degree)
        {
          case 1:
            project_matrix_free<components, 1>(mapping,
                                               dof,
                                               constraints,
                                               quadrature,
                                               function,
                                               work_result,
                                               enforce_zero_boundary,
                                               q_boundary,
                                               project_to_boundary_first);
            break;

          case 2:
            project_matrix_free<components, 2>(mapping,
                                               dof,
                                               constraints,
                                               quadrature,
                                               function,
                                               work_result,
                                               enforce_zero_boundary,
                                               q_boundary,
                                               project_to_boundary_first);
            break;

          case 3:
            project_matrix_free<components, 3>(mapping,
                                               dof,
                                               constraints,
                                               quadrature,
                                               function,
                                               work_result,
                                               enforce_zero_boundary,
                                               q_boundary,
                                               project_to_boundary_first);
            break;

          default:
            project_matrix_free<components, -1>(mapping,
                                                dof,
                                                constraints,
                                                quadrature,
                                                function,
                                                work_result,
                                                enforce_zero_boundary,
                                                q_boundary,
                                                project_to_boundary_first);
        }
    }



    // Helper interface for the matrix-free implementation of project().
    // Used to determine the number of components.
    template <int dim, typename Number, int spacedim>
    void
    project_matrix_free_component(
      const Mapping<dim, spacedim> &   mapping,
      const DoFHandler<dim, spacedim> &dof,
      const AffineConstraints<Number> &constraints,
      const Quadrature<dim> &          quadrature,
      const Function<
        spacedim,
        typename LinearAlgebra::distributed::Vector<Number>::value_type>
        &                                         function,
      LinearAlgebra::distributed::Vector<Number> &work_result,
      const bool                                  enforce_zero_boundary,
      const Quadrature<dim - 1> &                 q_boundary,
      const bool                                  project_to_boundary_first)
    {
      switch (dof.get_fe(0).n_components())
        {
          case 1:
            project_matrix_free_degree<1>(mapping,
                                          dof,
                                          constraints,
                                          quadrature,
                                          function,
                                          work_result,
                                          enforce_zero_boundary,
                                          q_boundary,
                                          project_to_boundary_first);
            break;

          case 2:
            project_matrix_free_degree<2>(mapping,
                                          dof,
                                          constraints,
                                          quadrature,
                                          function,
                                          work_result,
                                          enforce_zero_boundary,
                                          q_boundary,
                                          project_to_boundary_first);
            break;

          case 3:
            project_matrix_free_degree<3>(mapping,
                                          dof,
                                          constraints,
                                          quadrature,
                                          function,
                                          work_result,
                                          enforce_zero_boundary,
                                          q_boundary,
                                          project_to_boundary_first);
            break;

          case 4:
            project_matrix_free_degree<4>(mapping,
                                          dof,
                                          constraints,
                                          quadrature,
                                          function,
                                          work_result,
                                          enforce_zero_boundary,
                                          q_boundary,
                                          project_to_boundary_first);
            break;

          default:
            Assert(false, ExcInternalError());
        }
    }



    /**
     * Helper interface for the matrix-free implementation of project(): avoid
     * instantiating the other helper functions for more than one VectorType
     * by copying from a LinearAlgebra::distributed::Vector.
     */
    template <int dim, typename VectorType, int spacedim>
    void
    project_matrix_free_copy_vector(
      const Mapping<dim, spacedim> &                             mapping,
      const DoFHandler<dim, spacedim> &                          dof,
      const AffineConstraints<typename VectorType::value_type> & constraints,
      const Quadrature<dim> &                                    quadrature,
      const Function<spacedim, typename VectorType::value_type> &function,
      VectorType &                                               vec_result,
      const bool                 enforce_zero_boundary,
      const Quadrature<dim - 1> &q_boundary,
      const bool                 project_to_boundary_first)
    {
      Assert(vec_result.size() == dof.n_dofs(),
             ExcDimensionMismatch(vec_result.size(), dof.n_dofs()));

      LinearAlgebra::distributed::Vector<typename VectorType::value_type>
        work_result;
      project_matrix_free_component(mapping,
                                    dof,
                                    constraints,
                                    quadrature,
                                    function,
                                    work_result,
                                    enforce_zero_boundary,
                                    q_boundary,
                                    project_to_boundary_first);

      const IndexSet &          locally_owned_dofs = dof.locally_owned_dofs();
      IndexSet::ElementIterator it                 = locally_owned_dofs.begin();
      for (; it != locally_owned_dofs.end(); ++it)
        ::dealii::internal::ElementAccess<VectorType>::set(work_result(*it),
                                                           *it,
                                                           vec_result);
      vec_result.compress(VectorOperation::insert);
    }



    /**
     * Specialization of project() for the case dim==spacedim.
     * Check if we can use the MatrixFree implementation or need
     * to use the matrix based one.
     */
    template <typename VectorType, int dim>
    void
    project(
      const Mapping<dim> &                                      mapping,
      const DoFHandler<dim> &                                   dof,
      const AffineConstraints<typename VectorType::value_type> &constraints,
      const Quadrature<dim> &                                   quadrature,
      const Function<dim, typename VectorType::value_type> &    function,
      VectorType &                                              vec_result,
      const bool                 enforce_zero_boundary,
      const Quadrature<dim - 1> &q_boundary,
      const bool                 project_to_boundary_first)
    {
      // If we can, use the matrix-free implementation
      bool use_matrix_free =
        MatrixFree<dim, typename VectorType::value_type>::is_supported(
          dof.get_fe());

      // enforce_zero_boundary and project_to_boundary_first
      // are not yet supported.
      // We have explicit instantiations only if
      // the number of components is not too high.
      if (enforce_zero_boundary || project_to_boundary_first ||
          dof.get_fe(0).n_components() > 4)
        use_matrix_free = false;

      if (use_matrix_free)
        project_matrix_free_copy_vector(mapping,
                                        dof,
                                        constraints,
                                        quadrature,
                                        function,
                                        vec_result,
                                        enforce_zero_boundary,
                                        q_boundary,
                                        project_to_boundary_first);
      else
        {
          Assert((dynamic_cast<const parallel::TriangulationBase<dim> *>(
                    &(dof.get_triangulation())) == nullptr),
                 ExcNotImplemented());
          do_project(mapping,
                     dof,
                     constraints,
                     quadrature,
                     function,
                     vec_result,
                     enforce_zero_boundary,
                     q_boundary,
                     project_to_boundary_first);
        }
    }



    template <int dim, typename VectorType, int spacedim, int fe_degree>
    void
    project_parallel(
      const Mapping<dim, spacedim> &                            mapping,
      const DoFHandler<dim, spacedim> &                         dof,
      const AffineConstraints<typename VectorType::value_type> &constraints,
      const Quadrature<dim> &                                   quadrature,
      const std::function<typename VectorType::value_type(
        const typename DoFHandler<dim, spacedim>::active_cell_iterator &,
        const unsigned int)> &                                  func,
      VectorType &                                              vec_result)
    {
      using Number = typename VectorType::value_type;
      Assert(dof.get_fe(0).n_components() == 1,
             ExcDimensionMismatch(dof.get_fe(0).n_components(), 1));
      Assert(vec_result.size() == dof.n_dofs(),
             ExcDimensionMismatch(vec_result.size(), dof.n_dofs()));
      Assert(fe_degree == -1 ||
               dof.get_fe().degree == static_cast<unsigned int>(fe_degree),
             ExcDimensionMismatch(fe_degree, dof.get_fe().degree));

      // set up mass matrix and right hand side
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::partition_color;
      additional_data.mapping_update_flags =
        (update_values | update_JxW_values);
      std::shared_ptr<MatrixFree<dim, Number>> matrix_free(
        new MatrixFree<dim, Number>());
      matrix_free->reinit(mapping,
                          dof,
                          constraints,
                          QGauss<1>(dof.get_fe().degree + 2),
                          additional_data);
      using MatrixType = MatrixFreeOperators::MassOperator<
        dim,
        fe_degree,
        fe_degree + 2,
        1,
        LinearAlgebra::distributed::Vector<Number>>;
      MatrixType mass_matrix;
      mass_matrix.initialize(matrix_free);
      mass_matrix.compute_diagonal();

      using LocalVectorType = LinearAlgebra::distributed::Vector<Number>;
      LocalVectorType vec, rhs, inhomogeneities;
      matrix_free->initialize_dof_vector(vec);
      matrix_free->initialize_dof_vector(rhs);
      matrix_free->initialize_dof_vector(inhomogeneities);
      constraints.distribute(inhomogeneities);
      inhomogeneities *= -1.;

      // assemble right hand side:
      {
        FEValues<dim> fe_values(mapping,
                                dof.get_fe(),
                                quadrature,
                                update_values | update_JxW_values);

        const unsigned int dofs_per_cell = dof.get_fe().dofs_per_cell;
        const unsigned int n_q_points    = quadrature.size();
        Vector<Number>     cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        typename DoFHandler<dim, spacedim>::active_cell_iterator
          cell = dof.begin_active(),
          endc = dof.end();
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              cell_rhs = 0;
              fe_values.reinit(cell);
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  const double val_q = func(cell, q_point);
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    cell_rhs(i) += (fe_values.shape_value(i, q_point) * val_q *
                                    fe_values.JxW(q_point));
                }

              cell->get_dof_indices(local_dof_indices);
              constraints.distribute_local_to_global(cell_rhs,
                                                     local_dof_indices,
                                                     rhs);
            }
        rhs.compress(VectorOperation::add);
      }

      mass_matrix.vmult_add(rhs, inhomogeneities);

      // now invert the matrix
      // Allow for a maximum of 5*n steps to reduce the residual by 10^-12. n
      // steps may not be sufficient, since roundoff errors may accumulate for
      // badly conditioned matrices. This behavior can be observed, e.g. for
      // FE_Q_Hierarchical for degree higher than three.
      ReductionControl control(5 * rhs.size(), 0., 1e-12, false, false);
      SolverCG<LinearAlgebra::distributed::Vector<Number>>    cg(control);
      typename PreconditionJacobi<MatrixType>::AdditionalData data(0.8);
      PreconditionJacobi<MatrixType>                          preconditioner;
      preconditioner.initialize(mass_matrix, data);
      cg.solve(mass_matrix, vec, rhs, preconditioner);
      vec += inhomogeneities;

      constraints.distribute(vec);

      const IndexSet &          locally_owned_dofs = dof.locally_owned_dofs();
      IndexSet::ElementIterator it                 = locally_owned_dofs.begin();
      for (; it != locally_owned_dofs.end(); ++it)
        ::dealii::internal::ElementAccess<VectorType>::set(vec(*it),
                                                           *it,
                                                           vec_result);
      vec_result.compress(VectorOperation::insert);
    }



    template <int dim,
              typename VectorType,
              int spacedim,
              int fe_degree,
              int n_q_points_1d>
    void
    project_parallel(
      std::shared_ptr<const MatrixFree<dim, typename VectorType::value_type>>
                                                                matrix_free,
      const AffineConstraints<typename VectorType::value_type> &constraints,
      const std::function<VectorizedArray<typename VectorType::value_type>(
        const unsigned int,
        const unsigned int)> &                                  func,
      VectorType &                                              vec_result,
      const unsigned int                                        fe_component)
    {
      const DoFHandler<dim, spacedim> &dof =
        matrix_free->get_dof_handler(fe_component);

      using Number = typename VectorType::value_type;
      Assert(dof.get_fe(0).n_components() == 1,
             ExcDimensionMismatch(dof.get_fe(0).n_components(), 1));
      Assert(vec_result.size() == dof.n_dofs(),
             ExcDimensionMismatch(vec_result.size(), dof.n_dofs()));
      Assert(fe_degree == -1 ||
               dof.get_fe().degree == static_cast<unsigned int>(fe_degree),
             ExcDimensionMismatch(fe_degree, dof.get_fe().degree));

      using MatrixType = MatrixFreeOperators::MassOperator<
        dim,
        fe_degree,
        n_q_points_1d,
        1,
        LinearAlgebra::distributed::Vector<Number>>;
      MatrixType mass_matrix;
      mass_matrix.initialize(matrix_free, {fe_component});
      mass_matrix.compute_diagonal();

      using LocalVectorType = LinearAlgebra::distributed::Vector<Number>;
      LocalVectorType vec, rhs, inhomogeneities;
      matrix_free->initialize_dof_vector(vec, fe_component);
      matrix_free->initialize_dof_vector(rhs, fe_component);
      matrix_free->initialize_dof_vector(inhomogeneities, fe_component);
      constraints.distribute(inhomogeneities);
      inhomogeneities *= -1.;

      // assemble right hand side:
      {
        FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> fe_eval(
          *matrix_free, fe_component);
        const unsigned int n_cells    = matrix_free->n_macro_cells();
        const unsigned int n_q_points = fe_eval.n_q_points;

        for (unsigned int cell = 0; cell < n_cells; ++cell)
          {
            fe_eval.reinit(cell);
            for (unsigned int q = 0; q < n_q_points; ++q)
              fe_eval.submit_value(func(cell, q), q);

            fe_eval.integrate(true, false);
            fe_eval.distribute_local_to_global(rhs);
          }
        rhs.compress(VectorOperation::add);
      }

      mass_matrix.vmult_add(rhs, inhomogeneities);

      // now invert the matrix
      // Allow for a maximum of 5*n steps to reduce the residual by 10^-12. n
      // steps may not be sufficient, since roundoff errors may accumulate for
      // badly conditioned matrices. This behavior can be observed, e.g. for
      // FE_Q_Hierarchical for degree higher than three.
      ReductionControl control(5 * rhs.size(), 0., 1e-12, false, false);
      SolverCG<LinearAlgebra::distributed::Vector<Number>>    cg(control);
      typename PreconditionJacobi<MatrixType>::AdditionalData data(0.8);
      PreconditionJacobi<MatrixType>                          preconditioner;
      preconditioner.initialize(mass_matrix, data);
      cg.solve(mass_matrix, vec, rhs, preconditioner);
      vec += inhomogeneities;

      constraints.distribute(vec);

      const IndexSet &          locally_owned_dofs = dof.locally_owned_dofs();
      IndexSet::ElementIterator it                 = locally_owned_dofs.begin();
      for (; it != locally_owned_dofs.end(); ++it)
        ::dealii::internal::ElementAccess<VectorType>::set(vec(*it),
                                                           *it,
                                                           vec_result);
      vec_result.compress(VectorOperation::insert);
    }
  } // namespace internal



  template <int dim, typename VectorType, int spacedim>
  void
  project(const Mapping<dim, spacedim> &                            mapping,
          const DoFHandler<dim, spacedim> &                         dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const Quadrature<dim> &                                   quadrature,
          const std::function<typename VectorType::value_type(
            const typename DoFHandler<dim, spacedim>::active_cell_iterator &,
            const unsigned int)> &                                  func,
          VectorType &                                              vec_result)
  {
    switch (dof.get_fe().degree)
      {
        case 1:
          internal::project_parallel<dim, VectorType, spacedim, 1>(
            mapping, dof, constraints, quadrature, func, vec_result);
          break;
        case 2:
          internal::project_parallel<dim, VectorType, spacedim, 2>(
            mapping, dof, constraints, quadrature, func, vec_result);
          break;
        case 3:
          internal::project_parallel<dim, VectorType, spacedim, 3>(
            mapping, dof, constraints, quadrature, func, vec_result);
          break;
        default:
          internal::project_parallel<dim, VectorType, spacedim, -1>(
            mapping, dof, constraints, quadrature, func, vec_result);
      }
  }



  template <int dim, typename VectorType>
  void
  project(std::shared_ptr<const MatrixFree<
            dim,
            typename VectorType::value_type,
            VectorizedArray<typename VectorType::value_type>>>      matrix_free,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const unsigned int      n_q_points_1d,
          const std::function<VectorizedArray<typename VectorType::value_type>(
            const unsigned int,
            const unsigned int)> &func,
          VectorType &            vec_result,
          const unsigned int      fe_component)
  {
    const unsigned int fe_degree =
      matrix_free->get_dof_handler(fe_component).get_fe().degree;

    if (fe_degree + 1 == n_q_points_1d)
      switch (fe_degree)
        {
          case 1:
            internal::project_parallel<dim, VectorType, dim, 1, 2>(
              matrix_free, constraints, func, vec_result, fe_component);
            break;
          case 2:
            internal::project_parallel<dim, VectorType, dim, 2, 3>(
              matrix_free, constraints, func, vec_result, fe_component);
            break;
          case 3:
            internal::project_parallel<dim, VectorType, dim, 3, 4>(
              matrix_free, constraints, func, vec_result, fe_component);
            break;
          default:
            internal::project_parallel<dim, VectorType, dim, -1, 0>(
              matrix_free, constraints, func, vec_result, fe_component);
        }
    else
      internal::project_parallel<dim, VectorType, dim, -1, 0>(
        matrix_free, constraints, func, vec_result, fe_component);
  }



  template <int dim, typename VectorType>
  void
  project(std::shared_ptr<const MatrixFree<
            dim,
            typename VectorType::value_type,
            VectorizedArray<typename VectorType::value_type>>>      matrix_free,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const std::function<VectorizedArray<typename VectorType::value_type>(
            const unsigned int,
            const unsigned int)> &                                  func,
          VectorType &                                              vec_result,
          const unsigned int fe_component)
  {
    project(matrix_free,
            constraints,
            matrix_free->get_dof_handler(fe_component).get_fe().degree + 1,
            func,
            vec_result,
            fe_component);
  }



  template <int dim, typename VectorType, int spacedim>
  void
  project(const Mapping<dim, spacedim> &                            mapping,
          const DoFHandler<dim, spacedim> &                         dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const Quadrature<dim> &                                   quadrature,
          const Function<spacedim, typename VectorType::value_type> &function,
          VectorType &                                               vec_result,
          const bool                 enforce_zero_boundary,
          const Quadrature<dim - 1> &q_boundary,
          const bool                 project_to_boundary_first)
  {
    if (dim == spacedim)
      {
        const Mapping<dim> *const mapping_ptr =
          dynamic_cast<const Mapping<dim> *>(&mapping);
        const DoFHandler<dim> *const dof_ptr =
          dynamic_cast<const DoFHandler<dim> *>(&dof);
        const Function<dim,
                       typename VectorType::value_type> *const function_ptr =
          dynamic_cast<const Function<dim, typename VectorType::value_type> *>(
            &function);
        Assert(mapping_ptr != nullptr, ExcInternalError());
        Assert(dof_ptr != nullptr, ExcInternalError());
        internal::project<VectorType, dim>(*mapping_ptr,
                                           *dof_ptr,
                                           constraints,
                                           quadrature,
                                           *function_ptr,
                                           vec_result,
                                           enforce_zero_boundary,
                                           q_boundary,
                                           project_to_boundary_first);
      }
    else
      {
        Assert(
          (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
             &(dof.get_triangulation())) == nullptr),
          ExcNotImplemented());
        internal::do_project(mapping,
                             dof,
                             constraints,
                             quadrature,
                             function,
                             vec_result,
                             enforce_zero_boundary,
                             q_boundary,
                             project_to_boundary_first);
      }
  }



  template <int dim, typename VectorType, int spacedim>
  void
  project(const DoFHandler<dim, spacedim> &                         dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const Quadrature<dim> &                                   quadrature,
          const Function<spacedim, typename VectorType::value_type> &function,
          VectorType &                                               vec,
          const bool                 enforce_zero_boundary,
          const Quadrature<dim - 1> &q_boundary,
          const bool                 project_to_boundary_first)
  {
#ifdef _MSC_VER
    Assert(false,
           ExcMessage("Please specify the mapping explicitly "
                      "when building with MSVC!"));
#else
    project(StaticMappingQ1<dim, spacedim>::mapping,
            dof,
            constraints,
            quadrature,
            function,
            vec,
            enforce_zero_boundary,
            q_boundary,
            project_to_boundary_first);
#endif
  }



  template <int dim, typename VectorType, int spacedim>
  void
  project(const hp::MappingCollection<dim, spacedim> &              mapping,
          const hp::DoFHandler<dim, spacedim> &                     dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const hp::QCollection<dim> &                              quadrature,
          const Function<spacedim, typename VectorType::value_type> &function,
          VectorType &                                               vec_result,
          const bool                      enforce_zero_boundary,
          const hp::QCollection<dim - 1> &q_boundary,
          const bool                      project_to_boundary_first)
  {
    Assert((dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
              &(dof.get_triangulation())) == nullptr),
           ExcNotImplemented());

    internal::do_project(mapping,
                         dof,
                         constraints,
                         quadrature,
                         function,
                         vec_result,
                         enforce_zero_boundary,
                         q_boundary,
                         project_to_boundary_first);
  }


  template <int dim, typename VectorType, int spacedim>
  void
  project(const hp::DoFHandler<dim, spacedim> &                     dof,
          const AffineConstraints<typename VectorType::value_type> &constraints,
          const hp::QCollection<dim> &                              quadrature,
          const Function<spacedim, typename VectorType::value_type> &function,
          VectorType &                                               vec,
          const bool                      enforce_zero_boundary,
          const hp::QCollection<dim - 1> &q_boundary,
          const bool                      project_to_boundary_first)
  {
    project(hp::StaticMappingQ1<dim, spacedim>::mapping_collection,
            dof,
            constraints,
            quadrature,
            function,
            vec,
            enforce_zero_boundary,
            q_boundary,
            project_to_boundary_first);
  }



  template <int dim, int spacedim>
  void
  create_point_source_vector(const Mapping<dim, spacedim> &   mapping,
                             const DoFHandler<dim, spacedim> &dof_handler,
                             const Point<spacedim> &          p,
                             Vector<double> &                 rhs_vector)
  {
    Assert(rhs_vector.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(rhs_vector.size(), dof_handler.n_dofs()));
    Assert(dof_handler.get_fe(0).n_components() == 1,
           ExcMessage("This function only works for scalar finite elements"));

    rhs_vector = 0;

    std::pair<typename DoFHandler<dim, spacedim>::active_cell_iterator,
              Point<spacedim>>
      cell_point =
        GridTools::find_active_cell_around_point(mapping, dof_handler, p);

    Quadrature<dim> q(
      GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    FEValues<dim, spacedim> fe_values(mapping,
                                      dof_handler.get_fe(),
                                      q,
                                      UpdateFlags(update_values));
    fe_values.reinit(cell_point.first);

    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    cell_point.first->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; i++)
      rhs_vector(local_dof_indices[i]) = fe_values.shape_value(i, 0);
  }



  template <int dim, int spacedim>
  void
  create_point_source_vector(const DoFHandler<dim, spacedim> &dof_handler,
                             const Point<spacedim> &          p,
                             Vector<double> &                 rhs_vector)
  {
    create_point_source_vector(StaticMappingQ1<dim, spacedim>::mapping,
                               dof_handler,
                               p,
                               rhs_vector);
  }


  template <int dim, int spacedim>
  void
  create_point_source_vector(
    const hp::MappingCollection<dim, spacedim> &mapping,
    const hp::DoFHandler<dim, spacedim> &       dof_handler,
    const Point<spacedim> &                     p,
    Vector<double> &                            rhs_vector)
  {
    Assert(rhs_vector.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(rhs_vector.size(), dof_handler.n_dofs()));
    Assert(dof_handler.get_fe(0).n_components() == 1,
           ExcMessage("This function only works for scalar finite elements"));

    rhs_vector = 0;

    std::pair<typename hp::DoFHandler<dim, spacedim>::active_cell_iterator,
              Point<spacedim>>
      cell_point =
        GridTools::find_active_cell_around_point(mapping, dof_handler, p);

    Quadrature<dim> q(
      GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    FEValues<dim> fe_values(mapping[cell_point.first->active_fe_index()],
                            cell_point.first->get_fe(),
                            q,
                            UpdateFlags(update_values));
    fe_values.reinit(cell_point.first);

    const unsigned int dofs_per_cell = cell_point.first->get_fe().dofs_per_cell;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    cell_point.first->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; i++)
      rhs_vector(local_dof_indices[i]) = fe_values.shape_value(i, 0);
  }



  template <int dim, int spacedim>
  void
  create_point_source_vector(const hp::DoFHandler<dim, spacedim> &dof_handler,
                             const Point<spacedim> &              p,
                             Vector<double> &                     rhs_vector)
  {
    create_point_source_vector(hp::StaticMappingQ1<dim>::mapping_collection,
                               dof_handler,
                               p,
                               rhs_vector);
  }



  template <int dim, int spacedim>
  void
  create_point_source_vector(const Mapping<dim, spacedim> &   mapping,
                             const DoFHandler<dim, spacedim> &dof_handler,
                             const Point<spacedim> &          p,
                             const Point<dim> &               orientation,
                             Vector<double> &                 rhs_vector)
  {
    Assert(rhs_vector.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(rhs_vector.size(), dof_handler.n_dofs()));
    Assert(dof_handler.get_fe(0).n_components() == dim,
           ExcMessage(
             "This function only works for vector-valued finite elements."));

    rhs_vector = 0;

    const std::pair<typename DoFHandler<dim, spacedim>::active_cell_iterator,
                    Point<spacedim>>
      cell_point =
        GridTools::find_active_cell_around_point(mapping, dof_handler, p);

    const Quadrature<dim> q(
      GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    const FEValuesExtractors::Vector vec(0);
    FEValues<dim, spacedim>          fe_values(mapping,
                                      dof_handler.get_fe(),
                                      q,
                                      UpdateFlags(update_values));
    fe_values.reinit(cell_point.first);

    const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    cell_point.first->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; i++)
      rhs_vector(local_dof_indices[i]) =
        orientation * fe_values[vec].value(i, 0);
  }



  template <int dim, int spacedim>
  void
  create_point_source_vector(const DoFHandler<dim, spacedim> &dof_handler,
                             const Point<spacedim> &          p,
                             const Point<dim> &               orientation,
                             Vector<double> &                 rhs_vector)
  {
    create_point_source_vector(StaticMappingQ1<dim, spacedim>::mapping,
                               dof_handler,
                               p,
                               orientation,
                               rhs_vector);
  }


  template <int dim, int spacedim>
  void
  create_point_source_vector(
    const hp::MappingCollection<dim, spacedim> &mapping,
    const hp::DoFHandler<dim, spacedim> &       dof_handler,
    const Point<spacedim> &                     p,
    const Point<dim> &                          orientation,
    Vector<double> &                            rhs_vector)
  {
    Assert(rhs_vector.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(rhs_vector.size(), dof_handler.n_dofs()));
    Assert(dof_handler.get_fe(0).n_components() == dim,
           ExcMessage(
             "This function only works for vector-valued finite elements."));

    rhs_vector = 0;

    std::pair<typename hp::DoFHandler<dim, spacedim>::active_cell_iterator,
              Point<spacedim>>
      cell_point =
        GridTools::find_active_cell_around_point(mapping, dof_handler, p);

    Quadrature<dim> q(
      GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    const FEValuesExtractors::Vector vec(0);
    FEValues<dim> fe_values(mapping[cell_point.first->active_fe_index()],
                            cell_point.first->get_fe(),
                            q,
                            UpdateFlags(update_values));
    fe_values.reinit(cell_point.first);

    const unsigned int dofs_per_cell = cell_point.first->get_fe().dofs_per_cell;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    cell_point.first->get_dof_indices(local_dof_indices);

    for (unsigned int i = 0; i < dofs_per_cell; i++)
      rhs_vector(local_dof_indices[i]) =
        orientation * fe_values[vec].value(i, 0);
  }



  template <int dim, int spacedim>
  void
  create_point_source_vector(const hp::DoFHandler<dim, spacedim> &dof_handler,
                             const Point<spacedim> &              p,
                             const Point<dim> &                   orientation,
                             Vector<double> &                     rhs_vector)
  {
    create_point_source_vector(hp::StaticMappingQ1<dim>::mapping_collection,
                               dof_handler,
                               p,
                               orientation,
                               rhs_vector);
  }



  namespace internal
  {} // namespace internal



  namespace internals
  {} // namespace internals



  namespace internal
  {}



  template <int dim, typename VectorType, int spacedim>
  void
  point_difference(
    const DoFHandler<dim, spacedim> &                          dof,
    const VectorType &                                         fe_function,
    const Function<spacedim, typename VectorType::value_type> &exact_function,
    Vector<typename VectorType::value_type> &                  difference,
    const Point<spacedim> &                                    point)
  {
    point_difference(StaticMappingQ1<dim>::mapping,
                     dof,
                     fe_function,
                     exact_function,
                     difference,
                     point);
  }


  template <int dim, typename VectorType, int spacedim>
  void
  point_difference(
    const Mapping<dim, spacedim> &                             mapping,
    const DoFHandler<dim, spacedim> &                          dof,
    const VectorType &                                         fe_function,
    const Function<spacedim, typename VectorType::value_type> &exact_function,
    Vector<typename VectorType::value_type> &                  difference,
    const Point<spacedim> &                                    point)
  {
    using Number                 = typename VectorType::value_type;
    const FiniteElement<dim> &fe = dof.get_fe();

    Assert(difference.size() == fe.n_components(),
           ExcDimensionMismatch(difference.size(), fe.n_components()));

    // first find the cell in which this point
    // is, initialize a quadrature rule with
    // it, and then a FEValues object
    const std::pair<typename DoFHandler<dim, spacedim>::active_cell_iterator,
                    Point<spacedim>>
      cell_point =
        GridTools::find_active_cell_around_point(mapping, dof, point);

    AssertThrow(cell_point.first->is_locally_owned(),
                ExcPointNotAvailableHere());
    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-10,
           ExcInternalError());

    const Quadrature<dim> quadrature(
      GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
    FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
    fe_values.reinit(cell_point.first);

    // then use this to get at the values of
    // the given fe_function at this point
    std::vector<Vector<Number>> u_value(1, Vector<Number>(fe.n_components()));
    fe_values.get_function_values(fe_function, u_value);

    if (fe.n_components() == 1)
      difference(0) = exact_function.value(point);
    else
      exact_function.vector_value(point, difference);

    for (unsigned int i = 0; i < difference.size(); ++i)
      difference(i) -= u_value[0](i);
  }


  template <int dim, typename VectorType, int spacedim>
  void
  point_value(const DoFHandler<dim, spacedim> &        dof,
              const VectorType &                       fe_function,
              const Point<spacedim> &                  point,
              Vector<typename VectorType::value_type> &value)
  {
    point_value(
      StaticMappingQ1<dim, spacedim>::mapping, dof, fe_function, point, value);
  }


  template <int dim, typename VectorType, int spacedim>
  void
  point_value(const hp::DoFHandler<dim, spacedim> &    dof,
              const VectorType &                       fe_function,
              const Point<spacedim> &                  point,
              Vector<typename VectorType::value_type> &value)
  {
    point_value(hp::StaticMappingQ1<dim, spacedim>::mapping_collection,
                dof,
                fe_function,
                point,
                value);
  }


  template <int dim, typename VectorType, int spacedim>
  typename VectorType::value_type
  point_value(const DoFHandler<dim, spacedim> &dof,
              const VectorType &               fe_function,
              const Point<spacedim> &          point)
  {
    return point_value(StaticMappingQ1<dim, spacedim>::mapping,
                       dof,
                       fe_function,
                       point);
  }


  template <int dim, typename VectorType, int spacedim>
  typename VectorType::value_type
  point_value(const hp::DoFHandler<dim, spacedim> &dof,
              const VectorType &                   fe_function,
              const Point<spacedim> &              point)
  {
    return point_value(hp::StaticMappingQ1<dim, spacedim>::mapping_collection,
                       dof,
                       fe_function,
                       point);
  }


  template <int dim, typename VectorType, int spacedim>
  void
  point_value(const Mapping<dim, spacedim> &           mapping,
              const DoFHandler<dim, spacedim> &        dof,
              const VectorType &                       fe_function,
              const Point<spacedim> &                  point,
              Vector<typename VectorType::value_type> &value)
  {
    using Number                 = typename VectorType::value_type;
    const FiniteElement<dim> &fe = dof.get_fe();

    Assert(value.size() == fe.n_components(),
           ExcDimensionMismatch(value.size(), fe.n_components()));

    // first find the cell in which this point
    // is, initialize a quadrature rule with
    // it, and then a FEValues object
    const std::pair<typename DoFHandler<dim, spacedim>::active_cell_iterator,
                    Point<spacedim>>
      cell_point =
        GridTools::find_active_cell_around_point(mapping, dof, point);

    AssertThrow(cell_point.first->is_locally_owned(),
                ExcPointNotAvailableHere());
    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-10,
           ExcInternalError());

    const Quadrature<dim> quadrature(
      GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
    fe_values.reinit(cell_point.first);

    // then use this to get at the values of
    // the given fe_function at this point
    std::vector<Vector<Number>> u_value(1, Vector<Number>(fe.n_components()));
    fe_values.get_function_values(fe_function, u_value);

    value = u_value[0];
  }


  template <int dim, typename VectorType, int spacedim>
  void
  point_value(const hp::MappingCollection<dim, spacedim> &mapping,
              const hp::DoFHandler<dim, spacedim> &       dof,
              const VectorType &                          fe_function,
              const Point<spacedim> &                     point,
              Vector<typename VectorType::value_type> &   value)
  {
    using Number                              = typename VectorType::value_type;
    const hp::FECollection<dim, spacedim> &fe = dof.get_fe_collection();

    Assert(value.size() == fe.n_components(),
           ExcDimensionMismatch(value.size(), fe.n_components()));

    // first find the cell in which this point
    // is, initialize a quadrature rule with
    // it, and then a FEValues object
    const std::pair<
      typename hp::DoFHandler<dim, spacedim>::active_cell_iterator,
      Point<spacedim>>
      cell_point =
        GridTools::find_active_cell_around_point(mapping, dof, point);

    AssertThrow(cell_point.first->is_locally_owned(),
                ExcPointNotAvailableHere());
    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-10,
           ExcInternalError());

    const Quadrature<dim> quadrature(
      GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
    hp::FEValues<dim, spacedim> hp_fe_values(mapping,
                                             fe,
                                             hp::QCollection<dim>(quadrature),
                                             update_values);
    hp_fe_values.reinit(cell_point.first);
    const FEValues<dim, spacedim> &fe_values =
      hp_fe_values.get_present_fe_values();

    // then use this to get at the values of
    // the given fe_function at this point
    std::vector<Vector<Number>> u_value(1, Vector<Number>(fe.n_components()));
    fe_values.get_function_values(fe_function, u_value);

    value = u_value[0];
  }


  template <int dim, typename VectorType, int spacedim>
  typename VectorType::value_type
  point_value(const Mapping<dim, spacedim> &   mapping,
              const DoFHandler<dim, spacedim> &dof,
              const VectorType &               fe_function,
              const Point<spacedim> &          point)
  {
    Assert(dof.get_fe(0).n_components() == 1,
           ExcMessage(
             "Finite element is not scalar as is necessary for this function"));

    Vector<typename VectorType::value_type> value(1);
    point_value(mapping, dof, fe_function, point, value);

    return value(0);
  }


  template <int dim, typename VectorType, int spacedim>
  typename VectorType::value_type
  point_value(const hp::MappingCollection<dim, spacedim> &mapping,
              const hp::DoFHandler<dim, spacedim> &       dof,
              const VectorType &                          fe_function,
              const Point<spacedim> &                     point)
  {
    Assert(dof.get_fe(0).n_components() == 1,
           ExcMessage(
             "Finite element is not scalar as is necessary for this function"));

    Vector<typename VectorType::value_type> value(1);
    point_value(mapping, dof, fe_function, point, value);

    return value(0);
  }



  template <int dim,
            int spacedim,
            template <int, int> class DoFHandlerType,
            typename VectorType>
  void
  get_position_vector(const DoFHandlerType<dim, spacedim> &dh,
                      VectorType &                         vector,
                      const ComponentMask &                mask)
  {
    AssertDimension(vector.size(), dh.n_dofs());
    const FiniteElement<dim, spacedim> &fe = dh.get_fe();

    // Construct default fe_mask;
    const ComponentMask fe_mask(
      mask.size() ? mask :
                    ComponentMask(fe.get_nonzero_components(0).size(), true));

    AssertDimension(fe_mask.size(), fe.get_nonzero_components(0).size());

    std::vector<unsigned int> fe_to_real(fe_mask.size(),
                                         numbers::invalid_unsigned_int);
    unsigned int              size = 0;
    for (unsigned int i = 0; i < fe_mask.size(); ++i)
      {
        if (fe_mask[i])
          fe_to_real[i] = size++;
      }
    Assert(
      size == spacedim,
      ExcMessage(
        "The Component Mask you provided is invalid. It has to select exactly spacedim entries."));


    if (fe.has_support_points())
      {
        const Quadrature<dim> quad(fe.get_unit_support_points());

        MappingQGeneric<dim, spacedim> map_q(fe.degree);
        FEValues<dim, spacedim> fe_v(map_q, fe, quad, update_quadrature_points);
        std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);

        AssertDimension(fe.dofs_per_cell, fe.get_unit_support_points().size());
        Assert(fe.is_primitive(),
               ExcMessage("FE is not Primitive! This won't work."));

        for (const auto &cell : dh.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              fe_v.reinit(cell);
              cell->get_dof_indices(dofs);
              const std::vector<Point<spacedim>> &points =
                fe_v.get_quadrature_points();
              for (unsigned int q = 0; q < points.size(); ++q)
                {
                  const unsigned int comp =
                    fe.system_to_component_index(q).first;
                  if (fe_mask[comp])
                    ::dealii::internal::ElementAccess<VectorType>::set(
                      points[q][fe_to_real[comp]], dofs[q], vector);
                }
            }
      }
    else
      {
        // Construct a FiniteElement with FE_Q^spacedim, and call this
        // function again.
        //
        // Once we have this, interpolate with the given finite element
        // to get a Mapping which is interpolatory at the support points
        // of FE_Q(fe.degree())
        const FESystem<dim, spacedim> *fe_system =
          dynamic_cast<const FESystem<dim, spacedim> *>(&fe);
        Assert(fe_system, ExcNotImplemented());
        unsigned int degree = numbers::invalid_unsigned_int;

        // Get information about the blocks
        for (unsigned int i = 0; i < fe_mask.size(); ++i)
          if (fe_mask[i])
            {
              const unsigned int base_i =
                fe_system->component_to_base_index(i).first;
              Assert(degree == numbers::invalid_unsigned_int ||
                       degree == fe_system->base_element(base_i).degree,
                     ExcNotImplemented());
              Assert(fe_system->base_element(base_i).is_primitive(),
                     ExcNotImplemented());
              degree = fe_system->base_element(base_i).degree;
            }

        // We create an intermediate FE_Q vector space, and then
        // interpolate from that vector space to this one, by
        // carefully selecting the right components.

        FESystem<dim, spacedim> feq(FE_Q<dim, spacedim>(degree), spacedim);
        DoFHandlerType<dim, spacedim> dhq(dh.get_triangulation());
        dhq.distribute_dofs(feq);
        Vector<double>      eulerq(dhq.n_dofs());
        const ComponentMask maskq(spacedim, true);
        get_position_vector(dhq, eulerq);

        FullMatrix<double> transfer(fe.dofs_per_cell, feq.dofs_per_cell);
        FullMatrix<double> local_transfer(feq.dofs_per_cell);
        const std::vector<Point<dim>> &points = feq.get_unit_support_points();

        // Here we construct the interpolation matrix from
        // FE_Q^spacedim to the FiniteElement used by
        // euler_dof_handler.
        //
        // In order to construct such interpolation matrix, we have to
        // solve the following system:
        //
        // v_j phi_j(q_i) = w_k psi_k(q_i) = w_k delta_ki = w_i
        //
        // where psi_k are the basis functions for fe_q, and phi_i are
        // the basis functions of the target space while q_i are the
        // support points for the fe_q space. With this choice of
        // interpolation points, on the matrices is the identity
        // matrix, and we have to invert only one matrix. The
        // resulting vector will be interpolatory at the support
        // points of fe_q, even if the finite element does not have
        // support points.
        //
        // Morally, we should invert the matrix T_ij = phi_i(q_j),
        // however in general this matrix is not invertible, since
        // there may be components which do not contribute to the
        // displacement vector. Since we are not interested in those
        // components, we construct a square matrix with the same
        // number of components of the FE_Q system. The FE_Q system
        // was constructed above in such a way that the polynomial
        // degree of the FE_Q system and that of the given FE are the
        // same on the cell, which should guarantee that, for the
        // displacement components only, the interpolation matrix is
        // invertible. We construct a mapping between indices first,
        // and check that this is the case. If not, we bail out, not
        // knowing what to do in this case.

        std::vector<unsigned int> fe_to_feq(fe.dofs_per_cell,
                                            numbers::invalid_unsigned_int);
        unsigned int              index = 0;
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if (fe_mask[fe.system_to_component_index(i).first])
            fe_to_feq[i] = index++;

        // If index is not the same as feq.dofs_per_cell, we won't
        // know how to invert the resulting matrix. Bail out.
        Assert(index == feq.dofs_per_cell, ExcNotImplemented());

        for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
          {
            const unsigned int comp_j = fe.system_to_component_index(j).first;
            if (fe_mask[comp_j])
              for (unsigned int i = 0; i < points.size(); ++i)
                {
                  if (fe_to_real[comp_j] ==
                      feq.system_to_component_index(i).first)
                    local_transfer(i, fe_to_feq[j]) =
                      fe.shape_value(j, points[i]);
                }
          }

        // Now we construct the rectangular interpolation matrix. This
        // one is filled only with the information from the components
        // of the displacement. The rest is set to zero.
        local_transfer.invert(local_transfer);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if (fe_to_feq[i] != numbers::invalid_unsigned_int)
            for (unsigned int j = 0; j < feq.dofs_per_cell; ++j)
              transfer(i, j) = local_transfer(fe_to_feq[i], j);

        // The interpolation matrix is then passed to the
        // VectorTools::interpolate() function to generate the correct
        // interpolation.
        interpolate(dhq, dh, transfer, eulerq, vector);
      }
  }
} // namespace VectorTools

DEAL_II_NAMESPACE_CLOSE

#endif
