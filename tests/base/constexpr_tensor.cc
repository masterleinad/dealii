// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2018 by the deal.II authors
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

// Create and copy tensors in constexpr setting

#include <deal.II/base/tensor.h>

#include "../tests.h"

template <int rank, int dim, typename Number>
void
test_consexpr_tensor_constructors()
{
  constexpr dealii::Tensor<rank, dim, Number> A;
  constexpr dealii::Tensor<rank, dim, Number> B(A);
  constexpr dealii::Tensor<rank, dim, Number> C = A;
  deallog << " Tensor<" << rank << "," << dim << ">" << std::endl;
  deallog << A << std::endl;
  deallog << B << std::endl;
  deallog << C << std::endl;
}

int
main()
{
  initlog();

  deallog << "Cheching constexpr default constructor of Tensor<rank,dim,Number>"
          << std::endl;
  {
    dealii::LogStream::Prefix p("float");
    test_consexpr_tensor_constructors<0, 1, float>();
    test_consexpr_tensor_constructors<0, 2, float>();
    test_consexpr_tensor_constructors<0, 3, float>();
    test_consexpr_tensor_constructors<1, 1, float>();
    test_consexpr_tensor_constructors<1, 2, float>();
    test_consexpr_tensor_constructors<1, 3, float>();
    test_consexpr_tensor_constructors<2, 1, float>();
    test_consexpr_tensor_constructors<2, 2, float>();
    test_consexpr_tensor_constructors<2, 3, float>();
  }

  {
    dealii::LogStream::Prefix p("double");
    test_consexpr_tensor_constructors<0, 1, double>();
    test_consexpr_tensor_constructors<0, 2, double>();
    test_consexpr_tensor_constructors<0, 3, double>();
    test_consexpr_tensor_constructors<1, 1, double>();
    test_consexpr_tensor_constructors<1, 2, double>();
    test_consexpr_tensor_constructors<1, 3, double>();
    test_consexpr_tensor_constructors<2, 1, double>();
    test_consexpr_tensor_constructors<2, 2, double>();
    test_consexpr_tensor_constructors<2, 3, double>();
  }

  deallog << "OK" << std::endl;
  return 0;
}
