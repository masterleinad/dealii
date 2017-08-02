// ---------------------------------------------------------------------
//
// Copyright (C) 2017 by the deal.II authors
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


#ifndef dealii__matrix_free_evaluation_selector_h
#define dealii__matrix_free_evaluation_selector_h

#include <deal.II/matrix_free/evaluation_kernels.h>

DEAL_II_NAMESPACE_OPEN

template <int dim, int n_components, typename Number>
struct Default
{
  static inline void evaluate (const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                               VectorizedArray<Number> *values_dofs_actual[],
                               VectorizedArray<Number> *values_quad[],
                               VectorizedArray<Number> *gradients_quad[][dim],
                               VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2],
                               VectorizedArray<Number> *scratch_data,
                               const bool               evaluate_values,
                               const bool               evaluate_gradients,
                               const bool               evaluate_hessians)
  {
    internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_general,
             dim, -1, 0, n_components, Number>
             ::evaluate(shape_info, values_dofs_actual, values_quad,
                        gradients_quad, hessians_quad, scratch_data,
                        evaluate_values, evaluate_gradients, evaluate_hessians);
  }

  static inline void integrate (const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                                VectorizedArray<Number> *values_dofs_actual[],
                                VectorizedArray<Number> *values_quad[],
                                VectorizedArray<Number> *gradients_quad[][dim],
                                VectorizedArray<Number> *scratch_data,
                                const bool               integrate_values,
                                const bool               integrate_gradients)
  {
    internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_general,
             dim, -1, 0, n_components, Number>
             ::integrate(shape_info, values_dofs_actual, values_quad,
                         gradients_quad, scratch_data,
                         integrate_values, integrate_gradients);
  }
};

template<int DEPTH , int degree, int n_q_points_1d, int dim, int n_components, typename Number>
struct Factory : Default<dim, n_components, Number> {};

template<int n_q_points_1d, int dim, int n_components, typename Number>
struct Factory<0, 10, n_q_points_1d, dim, n_components, Number> : Default<dim, n_components, Number> {};

template<int degree, int dim, int n_components, typename Number>
struct Factory<1, degree, 11, dim, n_components, Number> : Default<dim, n_components, Number> {};

template<int degree, int n_q_points_1d, int dim, int n_components, typename Number>
struct Factory<0, degree, n_q_points_1d, dim, n_components, Number>
{
  static inline void evaluate (
    const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
    VectorizedArray<Number> *values_dofs_actual[],
    VectorizedArray<Number> *values_quad[],
    VectorizedArray<Number> *gradients_quad[][dim],
    VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2],
    VectorizedArray<Number> *scratch_data,
    const bool               evaluate_values,
    const bool               evaluate_gradients,
    const bool               evaluate_hessians)
  {
    const unsigned int runtime_degree = shape_info.fe_degree;
    if (runtime_degree == degree)
      Factory<1, degree, 0, dim, n_components, Number>::evaluate (shape_info, values_dofs_actual, values_quad, gradients_quad, hessians_quad, scratch_data, evaluate_values, evaluate_gradients, evaluate_hessians);
    else
      Factory<0, degree + 1, n_q_points_1d, dim, n_components, Number>::evaluate (shape_info, values_dofs_actual, values_quad, gradients_quad, hessians_quad, scratch_data, evaluate_values, evaluate_gradients, evaluate_hessians);
  }

  static inline void integrate (
    const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
    VectorizedArray<Number> *values_dofs_actual[],
    VectorizedArray<Number> *values_quad[],
    VectorizedArray<Number> *gradients_quad[][dim],
    VectorizedArray<Number> *scratch_data,
    const bool               integrate_values,
    const bool               integrate_gradients)
  {
    const unsigned int runtime_degree = shape_info.fe_degree;
    if (runtime_degree == degree)
      Factory<1, degree, 0, dim, n_components, Number>::integrate (shape_info, values_dofs_actual, values_quad, gradients_quad, scratch_data, integrate_values, integrate_gradients);
    else
      Factory<0, degree + 1, n_q_points_1d, dim, n_components, Number>::integrate (shape_info, values_dofs_actual, values_quad, gradients_quad, scratch_data, integrate_values, integrate_gradients);
  }
};

template<int degree, int n_q_points_1d, int dim, int n_components, typename Number>
struct Factory<1, degree, n_q_points_1d, dim, n_components, Number>
{
  static inline void evaluate (const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                               VectorizedArray<Number> *values_dofs_actual[],
                               VectorizedArray<Number> *values_quad[],
                               VectorizedArray<Number> *gradients_quad[][dim],
                               VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2],
                               VectorizedArray<Number> *scratch_data,
                               const bool               evaluate_values,
                               const bool               evaluate_gradients,
                               const bool               evaluate_hessians)
  {
    const unsigned int runtime_n_q_points_1d = shape_info.n_q_points_1d;
    if (runtime_n_q_points_1d == n_q_points_1d)
      {
        if (n_q_points_1d == degree+1)
          {
            if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_collocation)
              internal::FEEvaluationImplCollocation<dim, degree, n_components, Number>
              ::evaluate(shape_info, values_dofs_actual, values_quad,
                         gradients_quad, hessians_quad, scratch_data,
                         evaluate_values, evaluate_gradients, evaluate_hessians);
            else
              internal::FEEvaluationImplTransformToCollocation<dim, degree, n_components, Number>
              ::evaluate(shape_info, values_dofs_actual, values_quad,
                         gradients_quad, hessians_quad, scratch_data,
                         evaluate_values, evaluate_gradients, evaluate_hessians);
          }
        else
          internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_symmetric, dim, degree, n_q_points_1d, n_components, Number>
          ::evaluate(shape_info, values_dofs_actual, values_quad,
                     gradients_quad, hessians_quad, scratch_data,
                     evaluate_values, evaluate_gradients, evaluate_hessians);
      }
    else
      Factory<1, degree, n_q_points_1d + 1, dim, n_components, Number>::evaluate (shape_info, values_dofs_actual, values_quad,
          gradients_quad, hessians_quad, scratch_data,
          evaluate_values, evaluate_gradients, evaluate_hessians);
  }

  static inline void integrate (const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                                VectorizedArray<Number> *values_dofs_actual[],
                                VectorizedArray<Number> *values_quad[],
                                VectorizedArray<Number> *gradients_quad[][dim],
                                VectorizedArray<Number> *scratch_data,
                                const bool               integrate_values,
                                const bool               integrate_gradients)
  {
    const unsigned int runtime_n_q_points_1d = shape_info.n_q_points_1d;
    if (runtime_n_q_points_1d == n_q_points_1d)
      {
        if (n_q_points_1d == degree+1)
          {
            if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_collocation)
              internal::FEEvaluationImplCollocation<dim, degree, n_components, Number>
              ::integrate(shape_info, values_dofs_actual, values_quad,
                          gradients_quad, scratch_data,
                          integrate_values, integrate_gradients);
            else
              internal::FEEvaluationImplTransformToCollocation<dim, degree, n_components, Number>
              ::integrate(shape_info, values_dofs_actual, values_quad,
                          gradients_quad, scratch_data,
                          integrate_values, integrate_gradients);
          }
        else
          internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_symmetric, dim, degree, n_q_points_1d, n_components, Number>
          ::integrate(shape_info, values_dofs_actual, values_quad,
                      gradients_quad, scratch_data,
                      integrate_values, integrate_gradients);
      }
    else
      Factory<1, degree, n_q_points_1d + 1, dim, n_components, Number>::integrate (shape_info, values_dofs_actual, values_quad,
          gradients_quad, scratch_data,
          integrate_values, integrate_gradients);
  }
};

template<int dim, int n_components, typename Number>
void symmetric_selector_evaluate (const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                                  VectorizedArray<Number> *values_dofs_actual[],
                                  VectorizedArray<Number> *values_quad[],
                                  VectorizedArray<Number> *gradients_quad[][dim],
                                  VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2],
                                  VectorizedArray<Number> *scratch_data,
                                  const bool               evaluate_values,
                                  const bool               evaluate_gradients,
                                  const bool               evaluate_hessians)
{
  Assert(shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric||
         shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_collocation,
         ExcInternalError());
  Factory<0,0,0, dim, n_components, Number>::evaluate (shape_info, values_dofs_actual, values_quad, gradients_quad, hessians_quad, scratch_data, evaluate_values, evaluate_gradients, evaluate_hessians);
}


template<int dim, int n_components, typename Number>
void symmetric_selector_integrate (const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                                   VectorizedArray<Number> *values_dofs_actual[],
                                   VectorizedArray<Number> *values_quad[],
                                   VectorizedArray<Number> *gradients_quad[][dim],
                                   VectorizedArray<Number> *scratch_data,
                                   const bool               integrate_values,
                                   const bool               integrate_gradients)
{
  Assert(shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric||
         shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_collocation,
         ExcInternalError());
  Factory<0,0,0, dim, n_components, Number>::integrate (shape_info, values_dofs_actual, values_quad, gradients_quad, scratch_data, integrate_values, integrate_gradients);
}



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename Number>
struct SelectEvaluator
{
  static void evaluate(const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                       VectorizedArray<Number> *values_dofs_actual[],
                       VectorizedArray<Number> *values_quad[],
                       VectorizedArray<Number> *gradients_quad[][dim],
                       VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2],
                       VectorizedArray<Number> *scratch_data,
                       const bool               evaluate_values,
                       const bool               evaluate_gradients,
                       const bool               evaluate_hessians);

  static void integrate(const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                        VectorizedArray<Number> *values_dofs_actual[],
                        VectorizedArray<Number> *values_quad[],
                        VectorizedArray<Number> *gradients_quad[][dim],
                        VectorizedArray<Number> *scratch_data,
                        const bool               integrate_values,
                        const bool               integrate_gradients);
};

template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename Number>
inline
void
SelectEvaluator<dim, fe_degree, n_q_points_1d, n_components, Number>::evaluate
(const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
 VectorizedArray<Number> *values_dofs_actual[],
 VectorizedArray<Number> *values_quad[],
 VectorizedArray<Number> *gradients_quad[][dim],
 VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2],
 VectorizedArray<Number> *scratch_data,
 const bool               evaluate_values,
 const bool               evaluate_gradients,
 const bool               evaluate_hessians)
{
  // Run time selection of algorithm matching the element type. Since some
  // template combinations do not work for fe_degree==-1, create safe template
  // values for the calls below.

  Assert(fe_degree>=0  && n_q_points_1d>0, ExcInternalError());

  if (fe_degree+1 == n_q_points_1d &&
      shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_collocation)
    {
      internal::FEEvaluationImplCollocation<dim, fe_degree, n_components, Number>
      ::evaluate(shape_info, values_dofs_actual, values_quad,
                 gradients_quad, hessians_quad, scratch_data,
                 evaluate_values, evaluate_gradients, evaluate_hessians);
    }
  else if (fe_degree+1 == n_q_points_1d &&
           shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric)
    {
      internal::FEEvaluationImplTransformToCollocation<dim, fe_degree, n_components, Number>
      ::evaluate(shape_info, values_dofs_actual, values_quad,
                 gradients_quad, hessians_quad, scratch_data,
                 evaluate_values, evaluate_gradients, evaluate_hessians);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_symmetric,
               dim, fe_degree, n_q_points_1d, n_components, Number>
               ::evaluate(shape_info, values_dofs_actual, values_quad,
                          gradients_quad, hessians_quad, scratch_data,
                          evaluate_values, evaluate_gradients, evaluate_hessians);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_plus_dg0)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_symmetric_plus_dg0,
               dim, fe_degree, n_q_points_1d, n_components, Number>
               ::evaluate(shape_info, values_dofs_actual, values_quad,
                          gradients_quad, hessians_quad, scratch_data,
                          evaluate_values, evaluate_gradients, evaluate_hessians);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::truncated_tensor)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::truncated_tensor,
               dim, fe_degree, n_q_points_1d, n_components, Number>
               ::evaluate(shape_info, values_dofs_actual, values_quad,
                          gradients_quad, hessians_quad, scratch_data,
                          evaluate_values, evaluate_gradients, evaluate_hessians);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_general)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_general,
               dim, fe_degree, n_q_points_1d, n_components, Number>
               ::evaluate(shape_info, values_dofs_actual, values_quad,
                          gradients_quad, hessians_quad, scratch_data,
                          evaluate_values, evaluate_gradients, evaluate_hessians);
    }
  else
    AssertThrow(false, ExcNotImplemented());
}

template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename Number>
inline
void
SelectEvaluator<dim, fe_degree, n_q_points_1d, n_components, Number>::integrate
(const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
 VectorizedArray<Number> *values_dofs_actual[],
 VectorizedArray<Number> *values_quad[],
 VectorizedArray<Number> *gradients_quad[][dim],
 VectorizedArray<Number> *scratch_data,
 const bool               integrate_values,
 const bool               integrate_gradients)
{
  // Run time selection of algorithm matching the element type. Since some
  // template combinations do not work for fe_degree==-1, create safe template
  // values for the calls below.

  Assert(fe_degree>=0  && n_q_points_1d>0, ExcInternalError());

  if (fe_degree+1 == n_q_points_1d &&
      shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_collocation)
    {
      internal::FEEvaluationImplCollocation<dim, fe_degree, n_components, Number>
      ::integrate(shape_info, values_dofs_actual, values_quad,
                  gradients_quad, scratch_data,
                  integrate_values, integrate_gradients);
    }
  else if (fe_degree+1 == n_q_points_1d &&
           shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric)
    {
      internal::FEEvaluationImplTransformToCollocation<dim, fe_degree, n_components, Number>
      ::integrate(shape_info, values_dofs_actual, values_quad,
                  gradients_quad, scratch_data,
                  integrate_values, integrate_gradients);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_symmetric,
               dim, fe_degree, n_q_points_1d, n_components, Number>
               ::integrate(shape_info, values_dofs_actual, values_quad,
                           gradients_quad, scratch_data,
                           integrate_values, integrate_gradients);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_plus_dg0)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_symmetric_plus_dg0,
               dim, fe_degree, n_q_points_1d, n_components, Number>
               ::integrate(shape_info, values_dofs_actual, values_quad,
                           gradients_quad, scratch_data,
                           integrate_values, integrate_gradients);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::truncated_tensor)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::truncated_tensor,
               dim, fe_degree, n_q_points_1d, n_components, Number>
               ::integrate(shape_info, values_dofs_actual, values_quad,
                           gradients_quad, scratch_data,
                           integrate_values, integrate_gradients);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_general)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_general,
               dim, fe_degree, n_q_points_1d, n_components, Number>
               ::integrate(shape_info, values_dofs_actual, values_quad,
                           gradients_quad, scratch_data,
                           integrate_values, integrate_gradients);
    }
  else
    AssertThrow(false, ExcNotImplemented());
}


template <int dim, int n_q_points_1d, int n_components, typename Number>
struct SelectEvaluator<dim, -1, n_q_points_1d, n_components, Number>
{
  static void evaluate(const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                       VectorizedArray<Number> *values_dofs_actual[],
                       VectorizedArray<Number> *values_quad[],
                       VectorizedArray<Number> *gradients_quad[][dim],
                       VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2],
                       VectorizedArray<Number> *scratch_data,
                       const bool               evaluate_values,
                       const bool               evaluate_gradients,
                       const bool               evaluate_hessians);

  static void integrate(const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                        VectorizedArray<Number> *values_dofs_actual[],
                        VectorizedArray<Number> *values_quad[],
                        VectorizedArray<Number> *gradients_quad[][dim],
                        VectorizedArray<Number> *scratch_data,
                        const bool               integrate_values,
                        const bool               integrate_gradients);
};

template <int dim, int dummy, int n_components, typename Number>
inline
void
SelectEvaluator<dim, -1, dummy, n_components, Number>::evaluate
(const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
 VectorizedArray<Number> *values_dofs_actual[],
 VectorizedArray<Number> *values_quad[],
 VectorizedArray<Number> *gradients_quad[][dim],
 VectorizedArray<Number> *hessians_quad[][(dim*(dim+1))/2],
 VectorizedArray<Number> *scratch_data,
 const bool               evaluate_values,
 const bool               evaluate_gradients,
 const bool               evaluate_hessians)
{
  if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_plus_dg0)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_symmetric_plus_dg0,
               dim, -1, 0, n_components, Number>
               ::evaluate(shape_info, values_dofs_actual, values_quad,
                          gradients_quad, hessians_quad, scratch_data,
                          evaluate_values, evaluate_gradients, evaluate_hessians);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::truncated_tensor)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::truncated_tensor,
               dim, -1, 0, n_components, Number>
               ::evaluate(shape_info, values_dofs_actual, values_quad,
                          gradients_quad, hessians_quad, scratch_data,
                          evaluate_values, evaluate_gradients, evaluate_hessians);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_general)
    internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_general,
             dim, -1, 0, n_components, Number>
             ::evaluate(shape_info, values_dofs_actual, values_quad,
                        gradients_quad, hessians_quad, scratch_data,
                        evaluate_values, evaluate_gradients, evaluate_hessians);
  else
    symmetric_selector_evaluate<dim, n_components, Number>
    (shape_info, values_dofs_actual, values_quad,
     gradients_quad, hessians_quad, scratch_data,
     evaluate_values, evaluate_gradients, evaluate_hessians);
}

template <int dim, int dummy, int n_components, typename Number>
inline
void
SelectEvaluator<dim, -1, dummy, n_components, Number>::integrate
(const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
 VectorizedArray<Number> *values_dofs_actual[],
 VectorizedArray<Number> *values_quad[],
 VectorizedArray<Number> *gradients_quad[][dim],
 VectorizedArray<Number> *scratch_data,
 const bool               integrate_values,
 const bool               integrate_gradients)
{
  if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_symmetric_plus_dg0)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_symmetric_plus_dg0,
               dim, -1, 0, n_components, Number>
               ::integrate(shape_info, values_dofs_actual, values_quad,
                           gradients_quad, scratch_data,
                           integrate_values, integrate_gradients);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::truncated_tensor)
    {
      internal::FEEvaluationImpl<internal::MatrixFreeFunctions::truncated_tensor,
               dim, -1, 0, n_components, Number>
               ::integrate(shape_info, values_dofs_actual, values_quad,
                           gradients_quad, scratch_data,
                           integrate_values, integrate_gradients);
    }
  else if (shape_info.element_type == internal::MatrixFreeFunctions::tensor_general)
    internal::FEEvaluationImpl<internal::MatrixFreeFunctions::tensor_general,
             dim, -1, 0, n_components, Number>
             ::integrate(shape_info, values_dofs_actual, values_quad,
                         gradients_quad, scratch_data,
                         integrate_values, integrate_gradients);
  else
    symmetric_selector_integrate<dim, n_components, Number>
    (shape_info, values_dofs_actual, values_quad,
     gradients_quad, scratch_data,
     integrate_values, integrate_gradients);
}

DEAL_II_NAMESPACE_CLOSE

#endif
