// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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

// create and manipulate constexpr SymmetricTensor objects

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include "../tests.h"

template <int dim, typename Number>
void
test_symmetric_tensor()
{
  deallog << "*** Test constexpr SymmetricTensor functions, "
          << "dim = " << Utilities::to_string(dim) << std::endl;

  constexpr Number                                a = 1.0;
  constexpr Tensor<1, dim, Number>                v{};
  constexpr const SymmetricTensor<2, dim, Number> A(
    unit_symmetric_tensor<dim, Number>());
  constexpr const SymmetricTensor<2, dim, Number> B(
    unit_symmetric_tensor<dim, Number>());
  constexpr const Tensor<2, dim, Number> A_ns(
    unit_symmetric_tensor<dim, Number>());
  constexpr const SymmetricTensor<4, dim, Number> HH(
    identity_tensor<dim, Number>());

  constexpr const SymmetricTensor<2, dim, Number> C1 = A + B;
  constexpr const SymmetricTensor<2, dim, Number> C2 = A - B;
  constexpr const SymmetricTensor<2, dim, Number> C4 = a * A;
  constexpr const SymmetricTensor<2, dim, Number> C5 = A * a;
  constexpr const SymmetricTensor<2, dim, Number> C6 = A / a;

  constexpr const Number det_A = determinant(A);
  constexpr const Number tr_A  = trace(A);
  constexpr const Number I1_A  = first_invariant(A);
  constexpr const Number I2_A  = second_invariant(A);
  constexpr const Number I3_A  = third_invariant(A);

  constexpr const SymmetricTensor<2, dim, Number> A_inv = invert(A);
  constexpr const SymmetricTensor<2, dim, Number> A_T   = transpose(A);
  constexpr const SymmetricTensor<2, dim, Number> A_dev = deviator(A);

  constexpr const Number                 A_ddot_B_1 = A * B;
  constexpr const Number                 sp_A_B     = scalar_product(A, B);
  constexpr const Tensor<4, dim, Number> op_A_B     = outer_product(A, B);

  constexpr const Tensor<1, dim, Number> v3 = A * v;
  constexpr const Tensor<1, dim, Number> v4 = v * A;
  constexpr const Tensor<2, dim, Number> C7 = A * A_ns;
  constexpr const Tensor<2, dim, Number> C8 = A_ns * A;
}

constexpr SymmetricTensor<2, 2>
get_tensor_2()
{
  SymmetricTensor<2, 2> A;
  A[0][0] = 1.;
  A[1][1] = 3.;
  A[0][1] = -5.;
  return A;
}


constexpr SymmetricTensor<4, 2>
get_tensor_4()
{
  SymmetricTensor<4, 2> B;
  B[0][0][0][0] = 1.;
  B[1][1][1][1] = 2.5;
  B[0][1][0][1] = 0.2;
  return B;
}


int
main()
{
  initlog();

  {
    LogStream::Prefix p("float");
    test_symmetric_tensor<1, float>();
    test_symmetric_tensor<2, float>();
    test_symmetric_tensor<3, float>();
  }

  {
    LogStream::Prefix p("double");
    test_symmetric_tensor<1, double>();
    test_symmetric_tensor<2, double>();
    test_symmetric_tensor<3, double>();
  }

  constexpr const auto A = get_tensor_2();
  deallog << "SymmetricTensor<2,2> = " << A << std::endl;

  constexpr const auto B = get_tensor_4();
  deallog << "SymmetricTensor<4,2> = " << B << std::endl;

  {
    constexpr double a_init[3][3] = {{1., 0., 0.}, {2., 1., 0.}, {3., 2., 1.}};
    constexpr Tensor<2, 3>                dummy_a{a_init};
    constexpr const SymmetricTensor<2, 3> a              = symmetrize(dummy_a);
    constexpr const auto                  inverted       = invert(a);
    constexpr double                      ref_init[3][3] = {{0., -2., 2.},
                                       {-2., 5., -2.},
                                       {2., -2., 0.}};
    constexpr Tensor<2, 3>                dummy_ref{ref_init};
    constexpr const SymmetricTensor<2, 3> ref = symmetrize(dummy_ref);
    Assert(inverted == ref, ExcInternalError());
  }
  {
    constexpr double a_init[3][3] = {{1., 2., 3.}, {2., 1., 2.}, {3., 2., 1.}};
    constexpr Tensor<2, 3>                dummy_a{a_init};
    constexpr const SymmetricTensor<2, 3> a          = symmetrize(dummy_a);
    constexpr const auto                  transposed = transpose(a);
    Assert(transposed == a, ExcInternalError());
    constexpr const auto dummy   = scalar_product(a, a);
    constexpr const auto dummy_5 = a * a;
    constexpr const auto middle  = outer_product(a, a);
    constexpr const auto dummy_6 = contract3(a, middle, a);
  }

  deallog << "OK" << std::endl;
  return 0;
}
