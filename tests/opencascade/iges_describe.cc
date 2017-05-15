//-----------------------------------------------------------
//
//    Copyright (C) 2014 - 2015 by the deal.II authors
//
//    This file is subject to LGPL and may not be distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//-----------------------------------------------------------

// Read goteborg.iges and dump its topological structure to the
// logfile.

#include "../tests.h"
#include <fstream>
#include <deal.II/base/logstream.h>

#include <deal.II/opencascade/utilities.h>
#include <TopTools.hxx>
#include <TopoDS_Shape.hxx>
#include <Standard_Stream.hxx>

using namespace OpenCASCADE;

int main ()
{
  initlog();

  TopoDS_Shape sh = read_IGES(SOURCE_DIR "/iges_files/goteborg.iges");
  std::vector<TopoDS_Face> faces;
  std::vector<TopoDS_Edge> edges;
  std::vector<TopoDS_Vertex> vertices;

  extract_geometrical_shapes(sh, faces, edges, vertices);

  std::tuple<unsigned int, unsigned int, unsigned int>
  n = count_elements(sh);
  unsigned int nf=std::get<0>(n);
  unsigned int ne=std::get<1>(n);
  unsigned int nv=std::get<2>(n);

  deallog << "Shape contains " << std::get<0>(n) << " faces, "
          << std::get<1>(n) << " edges, and "
          << std::get<2>(n) << " vertices." << std::endl;

  if (nf != faces.size())
    deallog << "Error!" << std::endl;
  if (ne != edges.size())
    deallog << "Error!" << std::endl;
  if (nv != vertices.size())
    deallog << "Error!" << std::endl;

  return 0;
}

