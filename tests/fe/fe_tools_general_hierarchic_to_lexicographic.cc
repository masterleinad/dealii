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
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

// Test that the general renumbering function works correctly.

#include <deal.II/fe/fe_tools.h>

#include "../tests.h"

using namespace std;

template <int dim>
void
test_renumbering(const std::vector<unsigned int> &dpo)
{
  const std::vector<unsigned int> return_vertices =
    FETools::general_lexicographic_to_hierarchic<dim>(dpo);
  for (unsigned int i = 0; i < return_vertices.size(); ++i)
    deallog << i << ": " << return_vertices[i] << std::endl;
}

int
main()
{
  initlog();

  {
    deallog.push("0");
    constexpr int             dim = 0;
    std::vector<unsigned int> dpo{1};
    test_renumbering<dim>(dpo);
    deallog.pop();
  }

  {
    deallog.push("1");
    constexpr int             dim = 1;
    std::vector<unsigned int> dpo{2, 1};
    test_renumbering<dim>(dpo);
    deallog.pop();
  }

  {
    deallog.push("2");
    constexpr int             dim = 2;
    std::vector<unsigned int> dpo{4, 2, 1};
    test_renumbering<dim>(dpo);
    deallog.pop();
  }

  {
    deallog.push("3");
    constexpr int             dim = 3;
    std::vector<unsigned int> dpo{8, 4, 2, 1};
    test_renumbering<dim>(dpo);
    deallog.pop();
  }

  {
    deallog.push("3");
    constexpr int             dim = 3;
    std::vector<unsigned int> dpo{1, 1, 1, 1};
    test_renumbering<dim>(dpo);
    deallog.pop();
  }

  return 0;
}
