// ---------------------------------------------------------------------
//
// Copyright (C) 2001 - 2017 by the deal.II authors
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

// check GridTools::diameter for codim-1 meshes

#include "../tests.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

std::ofstream logfile("output");

template <int dim>
void
test1()
{
  // test 1: hypercube
  if(true)
    {
      Triangulation<dim, dim + 1> tria;
      GridGenerator::hyper_cube(tria);

      for(unsigned int i= 0; i < 2; ++i)
        {
          tria.refine_global(2);
          deallog << dim << "d, "
                  << "hypercube diameter, " << i * 2
                  << " refinements: " << GridTools::diameter(tria) << std::endl;
        }
    }
}

int
main()
{
  deallog << std::setprecision(4);
  logfile << std::setprecision(4);
  deallog.attach(logfile);

  test1<1>();
  test1<2>();

  return 0;
}
