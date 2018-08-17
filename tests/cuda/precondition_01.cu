// ---------------------------------------------------------------------
//
// Copyright (C) 2018 by the deal.II authors
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

// Check that dealii::SolverCG works with CUDAWrappers::SparseMatrix

#include <deal.II/base/cuda.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/lac/cuda_sparse_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>

#include <memory>

#include "../testmatrix.h"
#include "../tests.h"

template <typename Number>
cusparseStatus_t
cusparseXcsric02(cusparseHandle_t         handle,
                 int                      m,
                 int                      nnz,
                 const cusparseMatDescr_t descrA,
                 Number *                 csrValA_valM,
                 const int *              csrRowPtrA,
                 const int *              csrColIndA,
                 csric02Info_t            info,
                 cusparseSolvePolicy_t    policy,
                 void *                   pBuffer);

template <>
cusparseStatus_t
cusparseXcsric02<float>(cusparseHandle_t         handle,
                        int                      m,
                        int                      nnz,
                        const cusparseMatDescr_t descrA,
                        float *                  csrValA_valM,
                        const int *              csrRowPtrA,
                        const int *              csrColIndA,
                        csric02Info_t            info,
                        cusparseSolvePolicy_t    policy,
                        void *                   pBuffer)
{
  return cusparseScsric02(handle,
                          m,
                          nnz,
                          descrA,
                          csrValA_valM,
                          csrRowPtrA,
                          csrColIndA,
                          info,
                          policy,
                          pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsric02<double>(cusparseHandle_t         handle,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         double *                 csrValA_valM,
                         const int *              csrRowPtrA,
                         const int *              csrColIndA,
                         csric02Info_t            info,
                         cusparseSolvePolicy_t    policy,
                         void *                   pBuffer)
{
  return cusparseDcsric02(handle,
                          m,
                          nnz,
                          descrA,
                          csrValA_valM,
                          csrRowPtrA,
                          csrColIndA,
                          info,
                          policy,
                          pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsric02<cuComplex>(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            cuComplex *              csrValA_valM,
                            const int *              csrRowPtrA,
                            const int *              csrColIndA,
                            csric02Info_t            info,
                            cusparseSolvePolicy_t    policy,
                            void *                   pBuffer)
{
  return cusparseCcsric02(handle,
                          m,
                          nnz,
                          descrA,
                          csrValA_valM,
                          csrRowPtrA,
                          csrColIndA,
                          info,
                          policy,
                          pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsric02<cuDoubleComplex>(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      nnz,
                                  const cusparseMatDescr_t descrA,
                                  cuDoubleComplex *        csrValA_valM,
                                  const int *              csrRowPtrA,
                                  const int *              csrColIndA,
                                  csric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void *                   pBuffer)
{
  return cusparseZcsric02(handle,
                          m,
                          nnz,
                          descrA,
                          csrValA_valM,
                          csrRowPtrA,
                          csrColIndA,
                          info,
                          policy,
                          pBuffer);
}



template <typename Number>
cusparseStatus_t
cusparseXcsrsv2_solve(cusparseHandle_t         handle,
                      cusparseOperation_t      transA,
                      int                      m,
                      int                      nnz,
                      const Number *           alpha,
                      const cusparseMatDescr_t descra,
                      const Number *           csrValA,
                      const int *              csrRowPtrA,
                      const int *              csrColIndA,
                      csrsv2Info_t             info,
                      const Number *           x,
                      Number *                 y,
                      cusparseSolvePolicy_t    policy,
                      void *                   pBuffer);

template <>
cusparseStatus_t
cusparseXcsrsv2_solve<float>(cusparseHandle_t         handle,
                             cusparseOperation_t      transA,
                             int                      m,
                             int                      nnz,
                             const float *            alpha,
                             const cusparseMatDescr_t descra,
                             const float *            csrValA,
                             const int *              csrRowPtrA,
                             const int *              csrColIndA,
                             csrsv2Info_t             info,
                             const float *            x,
                             float *                  y,
                             cusparseSolvePolicy_t    policy,
                             void *                   pBuffer)
{
  return cusparseScsrsv2_solve(handle,
                               transA,
                               m,
                               nnz,
                               alpha,
                               descra,
                               csrValA,
                               csrRowPtrA,
                               csrColIndA,
                               info,
                               x,
                               y,
                               policy,
                               pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsrsv2_solve<double>(cusparseHandle_t         handle,
                              cusparseOperation_t      transA,
                              int                      m,
                              int                      nnz,
                              const double *           alpha,
                              const cusparseMatDescr_t descra,
                              const double *           csrValA,
                              const int *              csrRowPtrA,
                              const int *              csrColIndA,
                              csrsv2Info_t             info,
                              const double *           x,
                              double *                 y,
                              cusparseSolvePolicy_t    policy,
                              void *                   pBuffer)
{
  return cusparseDcsrsv2_solve(handle,
                               transA,
                               m,
                               nnz,
                               alpha,
                               descra,
                               csrValA,
                               csrRowPtrA,
                               csrColIndA,
                               info,
                               x,
                               y,
                               policy,
                               pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsrsv2_solve<cuComplex>(cusparseHandle_t         handle,
                                 cusparseOperation_t      transA,
                                 int                      m,
                                 int                      nnz,
                                 const cuComplex *        alpha,
                                 const cusparseMatDescr_t descra,
                                 const cuComplex *        csrValA,
                                 const int *              csrRowPtrA,
                                 const int *              csrColIndA,
                                 csrsv2Info_t             info,
                                 const cuComplex *        x,
                                 cuComplex *              y,
                                 cusparseSolvePolicy_t    policy,
                                 void *                   pBuffer)
{
  return cusparseCcsrsv2_solve(handle,
                               transA,
                               m,
                               nnz,
                               alpha,
                               descra,
                               csrValA,
                               csrRowPtrA,
                               csrColIndA,
                               info,
                               x,
                               y,
                               policy,
                               pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsrsv2_solve<cuDoubleComplex>(cusparseHandle_t         handle,
                                       cusparseOperation_t      transA,
                                       int                      m,
                                       int                      nnz,
                                       const cuDoubleComplex *  alpha,
                                       const cusparseMatDescr_t descra,
                                       const cuDoubleComplex *  csrValA,
                                       const int *              csrRowPtrA,
                                       const int *              csrColIndA,
                                       csrsv2Info_t             info,
                                       const cuDoubleComplex *  x,
                                       cuDoubleComplex *        y,
                                       cusparseSolvePolicy_t    policy,
                                       void *                   pBuffer)
{
  return cusparseZcsrsv2_solve(handle,
                               transA,
                               m,
                               nnz,
                               alpha,
                               descra,
                               csrValA,
                               csrRowPtrA,
                               csrColIndA,
                               info,
                               x,
                               y,
                               policy,
                               pBuffer);
}


template <typename Number>
cusparseStatus_t
cusparseXcsrsv2_analysis(cusparseHandle_t         handle,
                         cusparseOperation_t      transA,
                         int                      m,
                         int                      nnz,
                         const cusparseMatDescr_t descrA,
                         const Number *           csrValA,
                         const int *              csrRowPtrA,
                         const int *              csrColIndA,
                         csrsv2Info_t             info,
                         cusparseSolvePolicy_t    policy,
                         void *                   pBuffer);

template <>
cusparseStatus_t
cusparseXcsrsv2_analysis<float>(cusparseHandle_t         handle,
                                cusparseOperation_t      transA,
                                int                      m,
                                int                      nnz,
                                const cusparseMatDescr_t descrA,
                                const float *            csrValA,
                                const int *              csrRowPtrA,
                                const int *              csrColIndA,
                                csrsv2Info_t             info,
                                cusparseSolvePolicy_t    policy,
                                void *                   pBuffer)
{
  return cusparseScsrsv2_analysis(handle,
                                  transA,
                                  m,
                                  nnz,
                                  descrA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  info,
                                  policy,
                                  pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsrsv2_analysis<double>(cusparseHandle_t         handle,
                                 cusparseOperation_t      transA,
                                 int                      m,
                                 int                      nnz,
                                 const cusparseMatDescr_t descrA,
                                 const double *           csrValA,
                                 const int *              csrRowPtrA,
                                 const int *              csrColIndA,
                                 csrsv2Info_t             info,
                                 cusparseSolvePolicy_t    policy,
                                 void *                   pBuffer)
{
  return cusparseDcsrsv2_analysis(handle,
                                  transA,
                                  m,
                                  nnz,
                                  descrA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  info,
                                  policy,
                                  pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsrsv2_analysis<cuComplex>(cusparseHandle_t         handle,
                                    cusparseOperation_t      transA,
                                    int                      m,
                                    int                      nnz,
                                    const cusparseMatDescr_t descrA,
                                    const cuComplex *        csrValA,
                                    const int *              csrRowPtrA,
                                    const int *              csrColIndA,
                                    csrsv2Info_t             info,
                                    cusparseSolvePolicy_t    policy,
                                    void *                   pBuffer)
{
  return cusparseCcsrsv2_analysis(handle,
                                  transA,
                                  m,
                                  nnz,
                                  descrA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  info,
                                  policy,
                                  pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsrsv2_analysis<cuDoubleComplex>(cusparseHandle_t         handle,
                                          cusparseOperation_t      transA,
                                          int                      m,
                                          int                      nnz,
                                          const cusparseMatDescr_t descrA,
                                          const cuDoubleComplex *  csrValA,
                                          const int *              csrRowPtrA,
                                          const int *              csrColIndA,
                                          csrsv2Info_t             info,
                                          cusparseSolvePolicy_t    policy,
                                          void *                   pBuffer)
{
  return cusparseZcsrsv2_analysis(handle,
                                  transA,
                                  m,
                                  nnz,
                                  descrA,
                                  csrValA,
                                  csrRowPtrA,
                                  csrColIndA,
                                  info,
                                  policy,
                                  pBuffer);
}



template <typename Number>
cusparseStatus_t
cusparseXcsric02_analysis(cusparseHandle_t         handle,
                          int                      m,
                          int                      nnz,
                          const cusparseMatDescr_t descrA,
                          const Number *           csrValA,
                          const int *              csrRowPtrA,
                          const int *              csrColIndA,
                          csric02Info_t            info,
                          cusparseSolvePolicy_t    policy,
                          void *                   pBuffer);

template <>
cusparseStatus_t
cusparseXcsric02_analysis<float>(cusparseHandle_t         handle,
                                 int                      m,
                                 int                      nnz,
                                 const cusparseMatDescr_t descrA,
                                 const float *            csrValA,
                                 const int *              csrRowPtrA,
                                 const int *              csrColIndA,
                                 csric02Info_t            info,
                                 cusparseSolvePolicy_t    policy,
                                 void *                   pBuffer)
{
  return cusparseScsric02_analysis(handle,
                                   m,
                                   nnz,
                                   descrA,
                                   csrValA,
                                   csrRowPtrA,
                                   csrColIndA,
                                   info,
                                   policy,
                                   pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsric02_analysis<double>(cusparseHandle_t         handle,
                                  int                      m,
                                  int                      nnz,
                                  const cusparseMatDescr_t descrA,
                                  const double *           csrValA,
                                  const int *              csrRowPtrA,
                                  const int *              csrColIndA,
                                  csric02Info_t            info,
                                  cusparseSolvePolicy_t    policy,
                                  void *                   pBuffer)
{
  return cusparseDcsric02_analysis(handle,
                                   m,
                                   nnz,
                                   descrA,
                                   csrValA,
                                   csrRowPtrA,
                                   csrColIndA,
                                   info,
                                   policy,
                                   pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsric02_analysis<cuComplex>(cusparseHandle_t         handle,
                                     int                      m,
                                     int                      nnz,
                                     const cusparseMatDescr_t descrA,
                                     const cuComplex *        csrValA,
                                     const int *              csrRowPtrA,
                                     const int *              csrColIndA,
                                     csric02Info_t            info,
                                     cusparseSolvePolicy_t    policy,
                                     void *                   pBuffer)
{
  return cusparseCcsric02_analysis(handle,
                                   m,
                                   nnz,
                                   descrA,
                                   csrValA,
                                   csrRowPtrA,
                                   csrColIndA,
                                   info,
                                   policy,
                                   pBuffer);
}

template <>
cusparseStatus_t
cusparseXcsric02_analysis<cuDoubleComplex>(cusparseHandle_t         handle,
                                           int                      m,
                                           int                      nnz,
                                           const cusparseMatDescr_t descrA,
                                           const cuDoubleComplex *  csrValA,
                                           const int *              csrRowPtrA,
                                           const int *              csrColIndA,
                                           csric02Info_t            info,
                                           cusparseSolvePolicy_t    policy,
                                           void *                   pBuffer)
{
  return cusparseZcsric02_analysis(handle,
                                   m,
                                   nnz,
                                   descrA,
                                   csrValA,
                                   csrRowPtrA,
                                   csrColIndA,
                                   info,
                                   policy,
                                   pBuffer);
}


template <typename Number>
cusparseStatus_t
cusparseXcsrsv2_bufferSize(cusparseHandle_t         handle,
                           cusparseOperation_t      transA,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           Number *                 csrValA,
                           const int *              csrRowPtrA,
                           const int *              csrColIndA,
                           csrsv2Info_t             info,
                           int *                    pBufferSizeInBytes);

template <>
cusparseStatus_t
cusparseXcsrsv2_bufferSize<float>(cusparseHandle_t         handle,
                                  cusparseOperation_t      transA,
                                  int                      m,
                                  int                      nnz,
                                  const cusparseMatDescr_t descrA,
                                  float *                  csrValA,
                                  const int *              csrRowPtrA,
                                  const int *              csrColIndA,
                                  csrsv2Info_t             info,
                                  int *                    pBufferSizeInBytes)
{
  return cusparseScsrsv2_bufferSize(handle,
                                    transA,
                                    m,
                                    nnz,
                                    descrA,
                                    csrValA,
                                    csrRowPtrA,
                                    csrColIndA,
                                    info,
                                    pBufferSizeInBytes);
}

template <>
cusparseStatus_t
cusparseXcsrsv2_bufferSize<double>(cusparseHandle_t         handle,
                                   cusparseOperation_t      transA,
                                   int                      m,
                                   int                      nnz,
                                   const cusparseMatDescr_t descrA,
                                   double *                 csrValA,
                                   const int *              csrRowPtrA,
                                   const int *              csrColIndA,
                                   csrsv2Info_t             info,
                                   int *                    pBufferSizeInBytes)
{
  return cusparseDcsrsv2_bufferSize(handle,
                                    transA,
                                    m,
                                    nnz,
                                    descrA,
                                    csrValA,
                                    csrRowPtrA,
                                    csrColIndA,
                                    info,
                                    pBufferSizeInBytes);
}

template <>
cusparseStatus_t
cusparseXcsrsv2_bufferSize<cuComplex>(cusparseHandle_t         handle,
                                      cusparseOperation_t      transA,
                                      int                      m,
                                      int                      nnz,
                                      const cusparseMatDescr_t descrA,
                                      cuComplex *              csrValA,
                                      const int *              csrRowPtrA,
                                      const int *              csrColIndA,
                                      csrsv2Info_t             info,
                                      int *pBufferSizeInBytes)
{
  return cusparseCcsrsv2_bufferSize(handle,
                                    transA,
                                    m,
                                    nnz,
                                    descrA,
                                    csrValA,
                                    csrRowPtrA,
                                    csrColIndA,
                                    info,
                                    pBufferSizeInBytes);
}

template <>
cusparseStatus_t
cusparseXcsrsv2_bufferSize<cuDoubleComplex>(cusparseHandle_t         handle,
                                            cusparseOperation_t      transA,
                                            int                      m,
                                            int                      nnz,
                                            const cusparseMatDescr_t descrA,
                                            cuDoubleComplex *        csrValA,
                                            const int *              csrRowPtrA,
                                            const int *              csrColIndA,
                                            csrsv2Info_t             info,
                                            int *pBufferSizeInBytes)
{
  return cusparseZcsrsv2_bufferSize(handle,
                                    transA,
                                    m,
                                    nnz,
                                    descrA,
                                    csrValA,
                                    csrRowPtrA,
                                    csrColIndA,
                                    info,
                                    pBufferSizeInBytes);
}



template <typename Number>
cusparseStatus_t
cusparseXcsric02_bufferSize(cusparseHandle_t         handle,
                            int                      m,
                            int                      nnz,
                            const cusparseMatDescr_t descrA,
                            Number *                 csrValA,
                            const int *              csrRowPtrA,
                            const int *              csrColIndA,
                            csric02Info_t            info,
                            int *                    pBufferSizeInBytes);

template <>
cusparseStatus_t
cusparseXcsric02_bufferSize<float>(cusparseHandle_t         handle,
                                   int                      m,
                                   int                      nnz,
                                   const cusparseMatDescr_t descrA,
                                   float *                  csrValA,
                                   const int *              csrRowPtrA,
                                   const int *              csrColIndA,
                                   csric02Info_t            info,
                                   int *                    pBufferSizeInBytes)
{
  return cusparseScsric02_bufferSize(handle,
                                     m,
                                     nnz,
                                     descrA,
                                     csrValA,
                                     csrRowPtrA,
                                     csrColIndA,
                                     info,
                                     pBufferSizeInBytes);
}

template <>
cusparseStatus_t
cusparseXcsric02_bufferSize<double>(cusparseHandle_t         handle,
                                    int                      m,
                                    int                      nnz,
                                    const cusparseMatDescr_t descrA,
                                    double *                 csrValA,
                                    const int *              csrRowPtrA,
                                    const int *              csrColIndA,
                                    csric02Info_t            info,
                                    int *                    pBufferSizeInBytes)
{
  return cusparseDcsric02_bufferSize(handle,
                                     m,
                                     nnz,
                                     descrA,
                                     csrValA,
                                     csrRowPtrA,
                                     csrColIndA,
                                     info,
                                     pBufferSizeInBytes);
}

template <>
cusparseStatus_t
cusparseXcsric02_bufferSize<cuComplex>(cusparseHandle_t         handle,
                                       int                      m,
                                       int                      nnz,
                                       const cusparseMatDescr_t descrA,
                                       cuComplex *              csrValA,
                                       const int *              csrRowPtrA,
                                       const int *              csrColIndA,
                                       csric02Info_t            info,
                                       int *pBufferSizeInBytes)
{
  return cusparseCcsric02_bufferSize(handle,
                                     m,
                                     nnz,
                                     descrA,
                                     csrValA,
                                     csrRowPtrA,
                                     csrColIndA,
                                     info,
                                     pBufferSizeInBytes);
}

template <>
cusparseStatus_t
cusparseXcsric02_bufferSize<cuDoubleComplex>(cusparseHandle_t         handle,
                                             int                      m,
                                             int                      nnz,
                                             const cusparseMatDescr_t descrA,
                                             cuDoubleComplex *        csrValA,
                                             const int *   csrRowPtrA,
                                             const int *   csrColIndA,
                                             csric02Info_t info,
                                             int *         pBufferSizeInBytes)
{
  return cusparseZcsric02_bufferSize(handle,
                                     m,
                                     nnz,
                                     descrA,
                                     csrValA,
                                     csrRowPtrA,
                                     csrColIndA,
                                     info,
                                     pBufferSizeInBytes);
}



namespace
{
  template <typename Number>
  void
  delete_device_vector(Number *device_ptr) noexcept
  {
    const cudaError_t error_code = cudaFree(device_ptr);
    (void)error_code;
    AssertNothrow(error_code == cudaSuccess,
                  dealii::ExcCudaError(cudaGetErrorString(error_code)));
  }
  template <typename Number>
  Number *
  allocate_device_vector(const std::size_t size)
  {
    Number *device_ptr;
    Utilities::CUDA::malloc(device_ptr, size);
    return device_ptr;
  }
} // namespace

template <typename Number>
void
apply_preconditioner(const CUDAWrappers::SparseMatrix<Number> &A,
                     const cusparseHandle_t                    cusparse_handle,
                     LinearAlgebra::CUDAWrappers::Vector<Number> &      dst,
                     const LinearAlgebra::CUDAWrappers::Vector<Number> &src)
{
  const Number *const    src_dev    = src.get_values();
  Number *               dst_dev    = dst.get_values();
  const cusparseHandle_t handle = cusparse_handle;

  const auto               cusparse_matrix    = A.get_cusparse_matrix();
  Number *                 d_csrVal           = std::get<0>(cusparse_matrix);
  const int *const         A_row_ptr_dev      = std::get<2>(cusparse_matrix);
  const int *const         A_column_index_dev = std::get<1>(cusparse_matrix);
  const cusparseMatDescr_t mat_descr          = std::get<3>(cusparse_matrix);

  const unsigned int m   = A.m();
  const unsigned int nnz = A.n_nonzero_elements();

  AssertDimension(dst.size(), src.size());
  AssertDimension(A.m(), src.size());
  AssertDimension(A.n(), src.size());

  std::unique_ptr<Number[], void (*)(Number *)> tmp_dev(
    allocate_device_vector<Number>(dst.size()), delete_device_vector<Number>);

  // Suppose that A is m x m sparse matrix represented by CSR format,
  // Assumption:
  // - handle is already created by cusparseCreate(),
  // - (A_row_ptr_dev, A_column_index_dev, d_csrVal) is CSR of A on device
  // memory,
  // - src_dev is right hand side vector on device memory,
  // - dst_dev is solution vector on device memory.
  // - tmp_dev is intermediate result on device memory.

  cusparseMatDescr_t          descr_M = mat_descr;
  cusparseMatDescr_t          descr_L = mat_descr;
  csric02Info_t               info_M  = 0;
  csrsv2Info_t                info_L  = 0;
  csrsv2Info_t                info_Lt = 0;
  int                         BufferSize_M;
  int                         BufferSize_L;
  int                         BufferSize_Lt;
  int                         BufferSize;
  void *                      pBuffer = 0;
  int                         structural_zero;
  int                         numerical_zero;
  const double                alpha     = 1.;
  const cusparseSolvePolicy_t policy_M  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policy_L  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t   trans_L   = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseOperation_t   trans_Lt  = CUSPARSE_OPERATION_TRANSPOSE;

  cusparseStatus_t status;
  // step 1: create a descriptor which contains
  // - matrix M is base-1
  // - matrix L is base-1
  // - matrix L is lower triangular
  // - matrix L has non-unit diagonal
  status = cusparseCreateMatDescr(&descr_M);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());

  status = cusparseCreateMatDescr(&descr_L);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());

  // step 2: create a empty info structure
  // we need one info for csric02 and two info's for csrsv2
  status = cusparseCreateCsric02Info(&info_M);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseCreateCsrsv2Info(&info_L);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseCreateCsrsv2Info(&info_Lt);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());

  // step 3: query how much memory used in csric02 and csrsv2, and allocate the
  // buffer
  status = cusparseXcsric02_bufferSize(handle,
                                       m,
                                       nnz,
                                       descr_M,
                                       d_csrVal,
                                       A_row_ptr_dev,
                                       A_column_index_dev,
                                       info_M,
                                       &BufferSize_M);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseXcsrsv2_bufferSize(handle,
                                      trans_L,
                                      m,
                                      nnz,
                                      descr_L,
                                      d_csrVal,
                                      A_row_ptr_dev,
                                      A_column_index_dev,
                                      info_L,
                                      &BufferSize_L);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseXcsrsv2_bufferSize(handle,
                                      trans_Lt,
                                      m,
                                      nnz,
                                      descr_L,
                                      d_csrVal,
                                      A_row_ptr_dev,
                                      A_column_index_dev,
                                      info_Lt,
                                      &BufferSize_Lt);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());

  BufferSize = max(BufferSize_M, max(BufferSize_L, BufferSize_Lt));

  // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
  cudaError_t status_cuda = cudaMalloc((void **)&pBuffer, BufferSize);
  Assert(cudaSuccess == status_cuda, ExcInternalError());

  // step 4: perform analysis of incomplete Cholesky on M
  //         perform analysis of triangular solve on L
  //         perform analysis of triangular solve on L'
  // The lower triangular part of M has the same sparsity pattern as L, so
  // we can do analysis of csric02 and csrsv2 simultaneously.

  status = cusparseXcsric02_analysis(handle,
                                     m,
                                     nnz,
                                     descr_M,
                                     d_csrVal,
                                     A_row_ptr_dev,
                                     A_column_index_dev,
                                     info_M,
                                     policy_M,
                                     pBuffer);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseXcsric02_zeroPivot(handle, info_M, &structural_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status)
    {
      printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

  status = cusparseXcsrsv2_analysis(handle,
                                    trans_Lt,
                                    m,
                                    nnz,
                                    descr_L,
                                    d_csrVal,
                                    A_row_ptr_dev,
                                    A_column_index_dev,
                                    info_Lt,
                                    policy_Lt,
                                    pBuffer);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());

  status = cusparseXcsrsv2_analysis(handle,
                                    trans_L,
                                    m,
                                    nnz,
                                    descr_L,
                                    d_csrVal,
                                    A_row_ptr_dev,
                                    A_column_index_dev,
                                    info_L,
                                    policy_L,
                                    pBuffer);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());

  // step 5: M = L * L'
  status = cusparseXcsric02(handle,
                            m,
                            nnz,
                            descr_M,
                            d_csrVal,
                            A_row_ptr_dev,
                            A_column_index_dev,
                            info_M,
                            policy_M,
                            pBuffer);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseXcsric02_zeroPivot(handle, info_M, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status)
    {
      printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

  // step 6: solve L*z = x
  status = cusparseXcsrsv2_solve(handle,
                                 trans_L,
                                 m,
                                 nnz,
                                 &alpha,
                                 descr_L,
                                 d_csrVal,
                                 A_row_ptr_dev,
                                 A_column_index_dev,
                                 info_L,
                                 src_dev,
                                 tmp_dev.get(),
                                 policy_L,
                                 pBuffer);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());

  // step 7: solve L'*y = z
  status = cusparseXcsrsv2_solve(handle,
                                 trans_Lt,
                                 m,
                                 nnz,
                                 &alpha,
                                 descr_L,
                                 d_csrVal,
                                 A_row_ptr_dev,
                                 A_column_index_dev,
                                 info_Lt,
                                 tmp_dev.get(),
                                 dst_dev,
                                 policy_Lt,
                                 pBuffer);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());

  // step 6: free resources
  status_cuda = cudaFree(pBuffer);
  Assert(cudaSuccess == status_cuda, ExcInternalError());
  status = cusparseDestroyMatDescr(descr_M);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseDestroyMatDescr(descr_L);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseDestroyCsric02Info(info_M);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseDestroyCsrsv2Info(info_L);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseDestroyCsrsv2Info(info_Lt);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
  status = cusparseDestroy(handle);
  Assert(CUSPARSE_STATUS_SUCCESS == status, ExcInternalError());
}

void
test(Utilities::CUDA::Handle &cuda_handle)
{
  // Build the sparse matrix on the host
  const unsigned int   problem_size = 3;
  unsigned int         size         = (problem_size - 1) * (problem_size - 1);
  FDMatrix             testproblem(problem_size, problem_size);
  SparsityPattern      structure(size, size, 5);
  SparseMatrix<double> A;
  testproblem.five_point_structure(structure);
  structure.compress();
  A.reinit(structure);
  testproblem.five_point(A);
  A.print(std::cout);

  // Solve on the host
  PreconditionIdentity prec_no;
  SolverControl        control(100, 1.e-10);
  SolverCG<>           cg_host(control);
  Vector<double>       sol_host(size);
  Vector<double>       rhs_host(size);
  for (unsigned int i = 0; i < size; ++i)
    rhs_host[i] = static_cast<double>(i);
  cg_host.solve(A, sol_host, rhs_host, prec_no);

  // Solve on the device
  CUDAWrappers::SparseMatrix<double>          A_dev(cuda_handle, A);
  LinearAlgebra::CUDAWrappers::Vector<double> sol_dev(size);
  LinearAlgebra::CUDAWrappers::Vector<double> rhs_dev(size);
  LinearAlgebra::ReadWriteVector<double>      rw_vector(size);
  for (unsigned int i = 0; i < size; ++i)
    rw_vector[i] = static_cast<double>(i);
  rhs_dev.import(rw_vector, VectorOperation::insert);
  SolverCG<LinearAlgebra::CUDAWrappers::Vector<double>> cg_dev(control);
  cg_dev.solve(A_dev, sol_dev, rhs_dev, prec_no);

  A_dev.print(std::cout);
  A_dev.print_formatted(std::cout);
  apply_preconditioner(A_dev, cuda_handle.cusparse_handle, sol_dev, rhs_dev);

  // Check the result
  rw_vector.import(sol_dev, VectorOperation::insert);
  for (unsigned int i = 0; i < size; ++i)
    AssertThrow(std::fabs(rw_vector[i] - sol_host[i]) < 1e-8,
                ExcInternalError());
}

int
main()
{
  initlog();
  deallog.depth_console(0);

  Utilities::CUDA::Handle cuda_handle;
  test(cuda_handle);

  deallog << "OK" << std::endl;

  return 0;
}
