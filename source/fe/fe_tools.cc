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

#include <deal.II/fe/fe_tools.templates.h>

DEAL_II_NAMESPACE_OPEN

namespace FETools
{
  template <>
  std::vector<unsigned int>
  general_lexicographic_to_hierarchic<0>(const std::vector<unsigned int> &dpo)
  {
    AssertDimension(dpo.size(), 1);
    std::vector<unsigned int> return_vertices(dpo[0]);
    std::iota(return_vertices.begin(), return_vertices.end(), 0);
    return return_vertices;
  }

  template <>
  std::vector<unsigned int>
  general_lexicographic_to_hierarchic<1>(const std::vector<unsigned int> &dpo)
  {
    AssertDimension(dpo.size(), 2);
    std::vector<unsigned int> return_vertices(2 * dpo[0] + dpo[1]);
    std::iota(return_vertices.begin(), return_vertices.end(), 0);
    return return_vertices;
  }

  template <>
  std::vector<unsigned int>
  general_lexicographic_to_hierarchic<2>(const std::vector<unsigned int> &dpo)
  {
    AssertDimension(dpo.size(), 3);

    std::vector<unsigned int> return_vertices(4 * dpo[0] + 4 * dpo[1] + dpo[2]);
    std::array<std::array<unsigned int *, 3>, 3> object_array;
    object_array[0][0] = return_vertices.data();      // vertex 0
    object_array[1][0] = object_array[0][0] + dpo[0]; // vertex 1
    object_array[0][1] = object_array[1][0] + dpo[0]; // vertex 2
    object_array[1][1] = object_array[0][1] + dpo[0]; // vertex 3
    object_array[0][2] = object_array[1][1] + dpo[0]; // line 0
    object_array[1][2] = object_array[0][2] + dpo[1]; // line 1
    object_array[2][0] = object_array[1][2] + dpo[1]; // line 2
    object_array[2][1] = object_array[2][0] + dpo[1]; // line 3
    object_array[2][2] = object_array[2][1] + dpo[1]; // cell

    const unsigned int n_vertex_dofs_1d = std::sqrt(dpo[0] + .5);
    const unsigned int n_line_dofs_1d   = dpo[1] / n_vertex_dofs_1d;
    AssertDimension(dpo[2], n_line_dofs_1d * n_line_dofs_1d);
    const unsigned int n_dofs_1d = 2 * n_vertex_dofs_1d + n_line_dofs_1d;

    unsigned int index = 0;
    for (unsigned int j = 0; j < n_dofs_1d; ++j)
      for (unsigned int i = 0; i < n_dofs_1d; ++i)
        *(object_array[i / n_vertex_dofs_1d][j / n_vertex_dofs_1d]++) = index++;

    AssertDimension(object_array[0][0] - return_vertices.data(),
                    dpo[0]); // vertex 0
    AssertDimension(object_array[1][0] - object_array[0][0],
                    dpo[0]); // vertex 1
    AssertDimension(object_array[0][1] - object_array[1][0],
                    dpo[0]); // vertex 2
    AssertDimension(object_array[1][1] - object_array[0][1],
                    dpo[0]); // vertex 3
    AssertDimension(object_array[0][2] - object_array[1][1], dpo[1]); // line 0
    AssertDimension(object_array[1][2] - object_array[0][2], dpo[1]); // line 1
    AssertDimension(object_array[2][0] - object_array[1][2], dpo[1]); // line 2
    AssertDimension(object_array[2][1] - object_array[2][0], dpo[1]); // line 3
    AssertDimension(object_array[2][2] - object_array[2][1], dpo[2]); // cell

    return return_vertices;
  }

  template <>
  std::vector<unsigned int>
  general_lexicographic_to_hierarchic<3>(const std::vector<unsigned int> &dpo)
  {
    AssertDimension(dpo.size(), 4);

    std::vector<unsigned int> return_vertices(8 * dpo[0] + 12 * dpo[1] +
                                              6 * dpo[2] + dpo[3]);
    std::array<std::array<std::array<unsigned int *, 3>, 3>, 3> object_array;
    object_array[0][0][0] = return_vertices.data();         // vertex 0
    object_array[1][0][0] = object_array[0][0][0] + dpo[0]; // vertex 1
    object_array[0][1][0] = object_array[1][0][0] + dpo[0]; // vertex 2
    object_array[1][1][0] = object_array[0][1][0] + dpo[0]; // vertex 3
    object_array[0][0][1] = object_array[1][1][0] + dpo[0]; // vertex 4
    object_array[1][0][1] = object_array[0][0][1] + dpo[0]; // vertex 5
    object_array[0][1][1] = object_array[1][0][1] + dpo[0]; // vertex 6
    object_array[1][1][1] = object_array[0][1][1] + dpo[0]; // vertex 7
    object_array[0][2][0] = object_array[1][1][1] + dpo[0]; // line 0
    object_array[1][2][0] = object_array[0][2][0] + dpo[1]; // line 1
    object_array[2][0][0] = object_array[1][2][0] + dpo[1]; // line 2
    object_array[2][1][0] = object_array[2][0][0] + dpo[1]; // line 3
    object_array[0][2][1] = object_array[2][1][0] + dpo[1]; // line 4
    object_array[1][2][1] = object_array[0][2][1] + dpo[1]; // line 5
    object_array[2][0][1] = object_array[1][2][1] + dpo[1]; // line 6
    object_array[2][1][1] = object_array[2][0][1] + dpo[1]; // line 7
    object_array[0][0][2] = object_array[2][1][1] + dpo[1]; // line 8
    object_array[1][0][2] = object_array[0][0][2] + dpo[1]; // line 9
    object_array[0][1][2] = object_array[1][0][2] + dpo[1]; // line 10
    object_array[1][1][2] = object_array[0][1][2] + dpo[1]; // line 11
    object_array[0][2][2] = object_array[1][1][2] + dpo[1]; // face 0
    object_array[1][2][2] = object_array[0][2][2] + dpo[2]; // face 1
    object_array[2][0][2] = object_array[1][2][2] + dpo[2]; // face 2
    object_array[2][1][2] = object_array[2][0][2] + dpo[2]; // face 3
    object_array[2][2][0] = object_array[2][1][2] + dpo[2]; // face 4
    object_array[2][2][1] = object_array[2][2][0] + dpo[2]; // face 5
    object_array[2][2][2] = object_array[2][2][1] + dpo[2]; // cell

    const unsigned int n_vertex_dofs_1d = std::pow(dpo[0] + .5, 1. / 3.);
    const unsigned int n_line_dofs_1d =
      dpo[1] / (n_vertex_dofs_1d * n_vertex_dofs_1d);
    AssertDimension(dpo[2], n_line_dofs_1d * n_line_dofs_1d * n_vertex_dofs_1d);
    AssertDimension(dpo[3], n_line_dofs_1d * n_line_dofs_1d * n_line_dofs_1d);
    const unsigned int n_dofs_1d = 2 * n_vertex_dofs_1d + n_line_dofs_1d;

    unsigned int index = 0;
    for (unsigned int k = 0; k < n_dofs_1d; ++k)
      for (unsigned int j = 0; j < n_dofs_1d; ++j)
        for (unsigned int i = 0; i < n_dofs_1d; ++i)
          *(object_array[i / n_vertex_dofs_1d][j / n_vertex_dofs_1d]
                        [k / n_vertex_dofs_1d]++) = index++;

    AssertDimension(object_array[0][0][0] - return_vertices.data(),
                    dpo[0]); // vertex 0
    AssertDimension(object_array[1][0][0] - object_array[0][0][0],
                    dpo[0]); // vertex 1
    AssertDimension(object_array[0][1][0] - object_array[1][0][0],
                    dpo[0]); // vertex 2
    AssertDimension(object_array[1][1][0] - object_array[0][1][0],
                    dpo[0]); // vertex 3
    AssertDimension(object_array[0][0][1] - object_array[1][1][0],
                    dpo[0]); // vertex 4
    AssertDimension(object_array[1][0][1] - object_array[0][0][1],
                    dpo[0]); // vertex 5
    AssertDimension(object_array[0][1][1] - object_array[1][0][1],
                    dpo[0]); // vertex 6
    AssertDimension(object_array[1][1][1] - object_array[0][1][1],
                    dpo[0]); // vertex 7
    AssertDimension(object_array[0][2][0] - object_array[1][1][1],
                    dpo[1]); // line 0
    AssertDimension(object_array[1][2][0] - object_array[0][2][0],
                    dpo[1]); // line 1
    AssertDimension(object_array[2][0][0] - object_array[1][2][0],
                    dpo[1]); // line 2
    AssertDimension(object_array[2][1][0] - object_array[2][0][0],
                    dpo[1]); // line 3
    AssertDimension(object_array[0][2][1] - object_array[2][1][0],
                    dpo[1]); // line 4
    AssertDimension(object_array[1][2][1] - object_array[0][2][1],
                    dpo[1]); // line 5
    AssertDimension(object_array[2][0][1] - object_array[1][2][1],
                    dpo[1]); // line 6
    AssertDimension(object_array[2][1][1] - object_array[2][0][1],
                    dpo[1]); // line 7
    AssertDimension(object_array[0][0][2] - object_array[2][1][1],
                    dpo[1]); // line 8
    AssertDimension(object_array[1][0][2] - object_array[0][0][2],
                    dpo[1]); // line 9
    AssertDimension(object_array[0][1][2] - object_array[1][0][2],
                    dpo[1]); // line 10
    AssertDimension(object_array[1][1][2] - object_array[0][1][2],
                    dpo[1]); // line 11
    AssertDimension(object_array[0][2][2] - object_array[1][1][2],
                    dpo[2]); // face 0
    AssertDimension(object_array[1][2][2] - object_array[0][2][2],
                    dpo[2]); // face 1
    AssertDimension(object_array[2][0][2] - object_array[1][2][2],
                    dpo[2]); // face 2
    AssertDimension(object_array[2][1][2] - object_array[2][0][2],
                    dpo[2]); // face 3
    AssertDimension(object_array[2][2][0] - object_array[2][1][2],
                    dpo[2]); // face 4
    AssertDimension(object_array[2][2][1] - object_array[2][2][0],
                    dpo[2]); // face 5
    AssertDimension(object_array[2][2][2] - object_array[2][2][1],
                    dpo[3]); // cell

    return return_vertices;
  }
} // namespace FETools

/*-------------- Explicit Instantiations -------------------------------*/
#include "fe_tools.inst"

DEAL_II_NAMESPACE_CLOSE
