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

#include "../tests.h"

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/vectorization.h>

int main()
{
  initlog();

  const unsigned int dim = 3;

  SymmetricTensor<4,dim,VectorizedArray<double> > I;
  SymmetricTensor<2,dim,VectorizedArray<double> > A;
  SymmetricTensor<2,dim,VectorizedArray<double> > B;
  SymmetricTensor<2,dim,VectorizedArray<double> > C;

  // I^sym = 0.5(d_ik*d_jl + d_il*d_jk) -> I^sym : A = A^sym
  for (unsigned int i = 0; i < dim; i++)
    for (unsigned int j = 0; j < dim; j++)
      for (unsigned int k = 0; k < dim; k++)
        for (unsigned int l = 0; l < dim; l++)
          {
            I[i][j][k][l] = ( (i == k && j== l && i == l && j == k) ? make_vectorized_array(1.0) : ( (i == k && j== l) || (i == l && j == k) ? make_vectorized_array(0.5) : make_vectorized_array(0.0)));
            for (unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; v++)
              deallog << "I[" << i << "][" << j << "][" << k << "]["  << l << "][" << v << "]= " << I[i][j][k][l][v] << std::endl;
          }

  double counter = 0.0;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; v++)
        {
          A[i][j][v] = counter;
          counter += 1.0;
          deallog << "A[" << i << "][" << j << "][" << v << "]= " << A[i][j][v] << std::endl;
        }

//  double_contract(C, I, A);
  B = I*A;

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; ++v)
        {
          deallog << "B[" << i << "][" << j << "][" << v << "]= " << B[i][j][v] << std::endl;
          deallog << "C[" << i << "][" << j << "][" << v << "]= " << C[i][j][v] << std::endl;
        }

  B -= A;

  // Note that you cannot use B.norm() here even with something
  // like VectorizedArray<double> frob_norm = B.norm() -> Maybe a TODO?
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; ++v)
        if (std::fabs(B[i][j][v]) >1.e-20)
          deallog<< "Not OK: " << B[i][j][v] << std::endl;

  deallog << "OK" << std::endl;
}
