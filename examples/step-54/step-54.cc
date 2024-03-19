/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2014 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 *  Authors: Andrea Mola, Luca Heltai, 2014
 */


// @sect3{Include files}

// We start with including a bunch of files that we will use in the
// various parts of the program. Most of them have been discussed in
// previous tutorials already:
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <Bnd_Box.hxx>
#include <BRepBndLib.hxx>
#include <TopExp_Explorer.hxx> 
#include <TopoDS.hxx>
#  include <TopoDS_Face.hxx>  
#  include <BRep_Tool.hxx>
#include <GeomLProp_SLProps.hxx>
#include <ShapeAnalysis_Surface.hxx>

// These are the headers of the opencascade support classes and
// functions. Notice that these will contain sensible data only if you
// compiled your deal.II library with support for OpenCASCADE, i.e.,
// specifying <code>-DDEAL_II_WITH_OPENCASCADE=ON</code> and
// <code>-DOPENCASCADE_DIR=/path/to/your/opencascade/installation</code>
// when calling <code>cmake</code> during deal.II configuration.
#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>


// Finally, a few C++ standard header files
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

// We isolate the rest of the program in its own namespace
namespace Step54
{
  using namespace dealii;



  // @sect3{The TriangulationOnCAD class}

  // This is the main class. All it really does is store names for
  // input and output files, and a triangulation. It then provides
  // a function that generates such a triangulation from a coarse
  // mesh, using one of the strategies discussed in the introduction
  // and listed in the enumeration type at the top of the class.
  //
  // The member functions of this class are similar to what you can
  // find in most of the other tutorial programs in the setup stage of
  // the grid for the simulations.

  class TriangulationOnCAD
  {
  public:
    enum ProjectionType
    {
      NormalProjection       = 0,
      DirectionalProjection  = 1,
      NormalToMeshProjection = 2
    };


    TriangulationOnCAD(
      const std::string   &initial_mesh_filename,
      const std::string   &cad_file_name,
      const std::string   &output_filename,
      const ProjectionType surface_projection_kind = NormalProjection);

    void run();

  private:
    void read_domain();

    void refine_mesh();

    void output_results(const unsigned int cycle);

    Triangulation<3, 3> tria;

    const std::string initial_mesh_filename;
    const std::string cad_file_name;
    const std::string output_filename;

    const ProjectionType surface_projection_kind;
  };


  // @sect4{TriangulationOnCAD::TriangulationOnCAD}

  // The constructor of the TriangulationOnCAD class is very simple.
  // The input arguments are strings for the input and output file
  // names, and the enumeration type that determines which kind of
  // surface projector is used in the mesh refinement cycles (see
  // below for details).

  TriangulationOnCAD::TriangulationOnCAD(
    const std::string   &initial_mesh_filename,
    const std::string   &cad_file_name,
    const std::string   &output_filename,
    const ProjectionType surface_projection_kind)
    : initial_mesh_filename(initial_mesh_filename)
    , cad_file_name(cad_file_name)
    , output_filename(output_filename)
    , surface_projection_kind(surface_projection_kind)
  {}

  void snap_to_iges(Triangulation<3>& tria, TopoDS_Shape& shape, OpenCASCADE::NormalProjectionManifold<3, 3>& projector)
  {
    Kokkos::Timer timer;

 std::map<unsigned int, std::tuple<std::reference_wrapper<Point<3>>, std::vector<Tensor<1,3>>, std::reference_wrapper<Point<3>>>> vertex_map;
    std::array<Tensor< 1, 3>, GeometryInfo<3>::vertices_per_face> normal_at_vertex;
    for (const auto& cell: tria.active_cell_iterators())
      {
        for (const unsigned int i : cell->face_indices())
          {
            const auto& face = cell->face(i);
            if (face->at_boundary())
              {
                projector.get_normals_at_vertices(face, normal_at_vertex);
                for (unsigned j = 0; j < face->n_vertices(); ++j)
                  {
                    const unsigned int     vertex_index = face->vertex_index(j);
                    const auto& vertex_map_iterator = vertex_map.find(vertex_index);
                    auto normal = normal_at_vertex[j]/normal_at_vertex[j].norm();

                    int closest_index = (j==0)?1:0;
                    Point<3> closest_y_neighbor = face->vertex(closest_index);
                    closest_y_neighbor(1) = face->vertex(j)(1);
                    double distance = (closest_y_neighbor - face->vertex(j)).norm();
                    for (unsigned int k=0; k<face->n_vertices(); ++k)
                    {
                      if (k!=j) {
                        closest_y_neighbor = face->vertex(k);
                        closest_y_neighbor(1) = face->vertex(j)(1);
                        double candidate_distance = (closest_y_neighbor - face->vertex(j)).norm();
                        if (candidate_distance < distance) {
                          distance = candidate_distance;
                          closest_index = k;
                        }
                      }
                    }

                    if(vertex_map_iterator == vertex_map.end()) {
                      std::tuple<std::reference_wrapper<Point<3>>, std::vector<Tensor<1,3>>, std::reference_wrapper<Point<3>>> pair(face->vertex(j), {normal}, face->vertex(closest_index));
                      vertex_map.emplace(vertex_index, pair);
                    } else {
                      closest_y_neighbor = std::get<2>(vertex_map_iterator->second);
                      closest_y_neighbor(1) = face->vertex(j)(1);
                      if (distance < (closest_y_neighbor - face->vertex(j)).norm())
                        std::get<2>(vertex_map_iterator->second) = face->vertex(closest_index);
                      std::get<1>(vertex_map_iterator->second).push_back(normal);
                    }
                  }
              }
          }
      }

    std::cout << "create_map: " << timer.seconds() << std::endl;
    timer.reset();
    std::cout << vertex_map.size() << std::endl;

   for (const auto& boundary_vertex_iterator: vertex_map)
    {
      const auto& normals = std::get<1>(boundary_vertex_iterator.second);
      double minimum_product = 1;
//      std::cout << "vertex: " << boundary_vertex_iterator.first << std::endl;
      for (unsigned int i = 0; i < normals.size(); ++i) {
  //      std::cout << normals[i] << std::endl;
        for(unsigned int j=i+1; j < normals.size(); ++j) {
          auto product = normals[i] * normals[j];
          minimum_product = std::min(product, minimum_product);
        }
      }
      //std::cout << minimum_product << std::endl;
      if (minimum_product > .5) {
        auto& vertex = std::get<0>(boundary_vertex_iterator.second).get();
        auto proj = OpenCASCADE::closest_point(shape, vertex, 1);
        vertex(0) = proj(0);
        vertex(2) = proj(2);
      }
    }

      std::cout << "project points: " << timer.seconds() << std::endl;
    timer.reset();

  for (const auto& boundary_vertex_iterator: vertex_map)
    {
      const auto& normals = std::get<1>(boundary_vertex_iterator.second);
      double minimum_product = 1;
    //  std::cout << "vertex: " << boundary_vertex_iterator.first << std::endl;
      for (unsigned int i = 0; i < normals.size(); ++i) {
      //  std::cout << normals[i] << std::endl;
        for(unsigned int j=i+1; j < normals.size(); ++j) {
          auto product = normals[i] * normals[j];
          minimum_product = std::min(product, minimum_product);
        }
      }
      //std::cout << minimum_product << std::endl;
      if (minimum_product <= .5) {
        auto& vertex = std::get<0>(boundary_vertex_iterator.second).get();
        auto& proj = std::get<2>(boundary_vertex_iterator.second).get();
        vertex(0) = proj(0);
        vertex(2) = proj(2);
      }
    }
         std::cout << "move points: " << timer.seconds() << std::endl;
  }

  void TriangulationOnCAD::read_domain()
  {
    TopoDS_Shape bow_surface = OpenCASCADE::read_IGES(cad_file_name, 1e-3);
    //OpenCASCADE::write_STL(bow_surface, "new_shape.STL", .1);
 
    //TopoDS_Shape bow_surface = OpenCASCADE::read_STL(cad_file_name);
    //OpenCASCADE::write_IGES(bow_surface, "new_shape.iges");
    //std::abort();

    double Xmin, Ymin, Zmin, Xmax, Ymax, Zmax;
    Bnd_Box B;
    BRepBndLib::Add(bow_surface, B);
    B.Get(Xmin, Ymin, Zmin, Xmax, Ymax, Zmax);

    std::cout << Xmin << ' '<< Ymin << ' ' <<  Zmin << ' ' << Xmax << ' ' << Ymax << ' ' << Zmax << std::endl;

    Triangulation<3> tesselated_mesh;
    GridGenerator::subdivided_hyper_rectangle(tesselated_mesh, std::vector<unsigned>{40,40,40}, Point<3>{Xmin, Ymin, Zmin}, Point<3>{Xmax, Ymax, Zmax});

    GridOut grid_out;
    std::ofstream mesh_stream("tesselated_mesh.vtk");
    grid_out.write_vtk(tesselated_mesh, mesh_stream);

    FE_Q<3> fe(1); 
    DoFHandler<3> dof_handler(tesselated_mesh);
    dof_handler.distribute_dofs(fe);    

    Vector<double> solution(dof_handler.n_dofs());

{
    TopExp_Explorer exp;
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
    std::vector<gp_Pnt> Pproj(fe.dofs_per_cell);
    std::vector<gp_Pnt> tmp_proj(fe.dofs_per_cell);
    std::vector<gp_Dir> tmp_normal(fe.dofs_per_cell);
    std::vector<double> minDistance(fe.dofs_per_cell, 1e7);
    std::vector<double>       u(fe.dofs_per_cell);
    std::vector<double>       v(fe.dofs_per_cell);
    std::vector<int> dof_index_accesses(dof_handler.n_dofs(), 0);

    std::cout << tesselated_mesh.n_cells() << std::endl;
    GeomLProp_SLProps props(1, 1.e-6);

    for (const auto& cell: dof_handler.active_cell_iterators())
    {
      //std::cout << cell_no++ << std::endl;
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i=0; i<local_dof_indices.size(); ++i) {
        Pproj[i] = OpenCASCADE::point(cell->vertex(i));
        minDistance[i] = 1.e7;
      }
                   
      for (exp.Init(bow_surface, TopAbs_FACE); exp.More(); exp.Next())
      {
        TopoDS_Face face = TopoDS::Face(exp.Current());
        Handle(Geom_Surface) SurfToProj = BRep_Tool::Surface(face);
    
        ShapeAnalysis_Surface projector(SurfToProj);
        props.SetSurface(SurfToProj);
        gp_Pnt2d proj_params;
        for (unsigned int i=0; i<local_dof_indices.size(); ++i) {
          if(dof_index_accesses[local_dof_indices[i]] !=0) continue;
          proj_params = projector.NextValueOfUV(proj_params, OpenCASCADE::point(cell->vertex(i)), 1.e-5);
      
          SurfToProj->D0(proj_params.X(), proj_params.Y(), tmp_proj[i]);
          props.SetParameters(proj_params.X(), proj_params.Y());

          double distance = OpenCASCADE::point<3>(tmp_proj[i]).distance(cell->vertex(i));
       
          if (distance < minDistance[i] && props.IsNormalDefined())
          {
            minDistance[i] = distance;
            Pproj[i]       = tmp_proj[i];
            u[i]           = proj_params.X();
            v[i]           = proj_params.Y();
            //props.SetParameters(u[i], v[i]);
            //if(props.IsNormalDefined())
              tmp_normal[i] = props.Normal();
          }
        }
      }
      for (unsigned int i=0; i<local_dof_indices.size(); ++i) {
        if(dof_index_accesses[local_dof_indices[i]] !=0) continue;
        Tensor<1, 3> dealii_normal({tmp_normal[i].X(), tmp_normal[i].Y(), tmp_normal[i].Z()});
        Point<3> dealii_projection(Pproj[i].X(), Pproj[i].Y(), Pproj[i].Z());
        double product = dealii_normal * (cell->vertex(i) - dealii_projection);
        solution(local_dof_indices[i]) = product;
        ++dof_index_accesses[local_dof_indices[i]];
      }
    }
}

  DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  const std::string filename = "solution.vtk";
  std::ofstream     output(filename);
  data_out.write_vtk(output);

    abort();

    // Each CAD geometrical object is defined along with a tolerance,
    // which indicates possible inaccuracy of its placement. For
    // instance, the tolerance @p tol of a vertex indicates that it can
    // be located in any point contained in a sphere centered in the
    // nominal position and having radius @p tol. While projecting a
    // point onto a surface (which will in turn have its tolerance) we
    // must keep in mind that the precision of the projection will be
    // limited by the tolerance with which the surface is built.

    // The following method extracts the tolerance of the given shape and
    // makes it a bit bigger to stay our of trouble:
    const double tolerance = 0.01;//OpenCASCADE::get_shape_tolerance(bow_surface) * 1000;

    // We now want to extract a set of composite sub-shapes from the
    // generic shape. In particular, each face of the CAD file
    // is composed of a trimming curve of type @p TopoDS_Wire, which is
    // the collection of @p TopoDS_Edges that compose the boundary of a
    // surface, and a NURBS description of the surface itself. We will
    // use a line projector to associate the boundary of our
    // Triangulation to the wire delimiting the surface.  To extract
    // all compound sub-shapes, like wires, shells, or solids, we
    // resort to a method of the OpenCASCADE namespace.  The input of
    // OpenCASCADE::extract_compound_shapes is a shape and a set of empty
    // std::vectors of subshapes, which will be filled with all
    // compound shapes found in the given topological shape:
    std::vector<TopoDS_Compound>  compounds;
    std::vector<TopoDS_CompSolid> compsolids;
    std::vector<TopoDS_Solid>     solids;
    std::vector<TopoDS_Shell>     shells;
    std::vector<TopoDS_Wire>      wires;

    OpenCASCADE::extract_compound_shapes(
      bow_surface, compounds, compsolids, solids, shells, wires);

    std::cout << "compounds:  " << compounds.size()   << std::endl;
    std::cout << "compsolids: " << compsolids.size()  << std::endl;
    std::cout << "solids:     " << solids.size()      << std::endl;
    std::cout << "shells:     " << shells.size()      << std::endl;
    std::cout << "wires:      " << wires.size()       << std::endl;


    // The next few steps are more familiar, and allow us to import an existing
    // mesh from an external VTK file, and convert it to a deal triangulation.
    std::ifstream in;

    in.open(initial_mesh_filename);

    GridIn<3, 3> gi;
    gi.attach_triangulation(tria);
    gi.read_vtk(in);
    GridTools::transform ([](const Point<3> &p) -> Point<3>
                      {
                        Point<3> q = p;
                        std::swap(q[1], q[2]);
                        q[0] -= 0.07;
                        q[2] -= 0.07;
                        return q;
                      },
                      tria);
    //GridTools::scale(.001, tria);
    //tria.refine_global(1);    

    // We output this initial mesh saving it as the refinement step 0.
    output_results(0);

    OpenCASCADE::NormalProjectionManifold<3, 3> normal_projector(
              bow_surface, 0.001);

   snap_to_iges(tria, bow_surface, normal_projector);
    output_results(1);

   for (const auto& cell: tria.active_cell_iterators())
      { if (cell->at_boundary())
     cell->set_refine_flag();
   }

   Kokkos::Timer timer;

   tria.execute_coarsening_and_refinement();
   std::cout << "Adaptive mesh refinement: " << timer.seconds() << std::endl;

   snap_to_iges(tria, bow_surface, normal_projector);
    output_results(2);

   abort();

 for (auto& cell: tria.active_cell_iterators())
    {
      for (int i =0; i<6; ++i) {
        if(cell->at_boundary(i)) {
          //if(std::abs(cell->center()(1)) < 0.036) {
            cell->face(i)->set_all_manifold_ids(1);
          //}
          //std::cout << "first vertex: " << cell->face(i)->vertex(0) << std::endl;

          for (int j=0; j<4; ++j) {
            auto proj = OpenCASCADE::closest_point(bow_surface, cell->face(i)->vertex(j), 1);
            //auto distance  = cell->face(i)->vertex(j).distance(proj);
            //if (distance > 0 && std::abs(cell->face(i)->vertex(j)(1)-.05) > 0.01) {
              //std::cout << "point: " << cell->face(i)->vertex(j) << " closest: " << proj << " distance: " << distance << std::endl;
              cell->face(i)->vertex(j) = proj;
            //}
          }
        }
      }
    }

    // Once both the CAD geometry and the initial mesh have been
    // imported and digested, we use the CAD surfaces and curves to
    // define the projectors and assign them to the manifold ids just
    // specified.

    // A first projector is defined using the single wire contained in
    // our CAD file.  The ArclengthProjectionLineManifold will make
    // sure that every mesh edge located on the wire is refined with a
    // point that lies on the wire and splits it into two equal arcs
    // lying between the edge vertices. We first check
    // that the wires vector contains at least one element and then
    // create a Manifold object for it.
    //
    // Once the projector is created, we then assign it to all the parts of
    // the triangulation with manifold_id = 2:
    Assert(
      wires.size() > 0,
      ExcMessage(
        "I could not find any wire in the CAD file you gave me. Bailing out."));

    OpenCASCADE::ArclengthProjectionLineManifold<3, 3> line_projector(
      wires[0], tolerance);

    //tria.set_manifold(2, line_projector);

    // The surface projector is created according to what is specified
    // with the @p surface_projection_kind option of the constructor. In particular,
    // if the surface_projection_kind value equals @p NormalProjection, we select the
    // OpenCASCADE::NormalProjectionManifold. The new mesh points will
    // then initially be generated at the barycenter of the cell/edge
    // considered, and then projected on the CAD surface along its
    // normal direction.  The NormalProjectionManifold constructor
    // only needs a shape and a tolerance, and we then assign it to
    // the triangulation for use with all parts that manifold having id 1:
    switch (surface_projection_kind)
      {
        case NormalProjection:
          {
            OpenCASCADE::NormalProjectionManifold<3, 3> normal_projector(
              bow_surface, tolerance);
            tria.set_manifold(1, normal_projector);

            break;
          }

        // @p If surface_projection_kind value is @p DirectionalProjection, we select the
        // OpenCASCADE::DirectionalProjectionManifold class. The new mesh points
        // will then initially be generated at the barycenter of the cell/edge
        // considered, and then projected on the CAD surface along a
        // direction that is specified to the
        // OpenCASCADE::DirectionalProjectionManifold constructor. In this case,
        // the projection is done along the y-axis.
        case DirectionalProjection:
          {
            OpenCASCADE::DirectionalProjectionManifold<3, 3>
              directional_projector(bow_surface,
                                    Point<3>(0.0, 1.0, 0.0),
                                    tolerance);
            tria.set_manifold(1, directional_projector);

            break;
          }

        // As a third option, if @p surface_projection_kind value
        // is @p NormalToMeshProjection, we select the
        // OpenCASCADE::NormalToMeshProjectionManifold. The new mesh points will
        // again initially be generated at the barycenter of the cell/edge
        // considered, and then projected on the CAD surface along a
        // direction that is an estimate of the mesh normal direction.
        // The OpenCASCADE::NormalToMeshProjectionManifold constructor only
        // requires a shape (containing at least a face) and a
        // tolerance.
        case NormalToMeshProjection:
          {
            OpenCASCADE::NormalToMeshProjectionManifold<3, 3>
              normal_to_mesh_projector(bow_surface, tolerance);
            tria.set_manifold(1, normal_to_mesh_projector);

            break;
          }

        // Finally, we use good software cleanliness by ensuring that this
        // really covers all possible options of the @p case statement. If we
        // get any other value, we simply abort the program:
        default:
          AssertThrow(false, ExcInternalError());
      }
  }


  // @sect4{TriangulationOnCAD::refine_mesh}

  // This function globally refines the mesh. In other tutorials, it
  // would typically also distribute degrees of freedom, and resize
  // matrices and vectors. These tasks are not carried out here, since
  // we are not running any simulation on the Triangulation produced.
  //
  // While the function looks innocent, this is where most of the work we are
  // interested in for this tutorial program actually happens. In particular,
  // when refining the quads and lines that define the surface of the ship's
  // hull, the Triangulation class will ask the various objects we have
  // assigned to handle individual manifold ids for where the new vertices
  // should lie.
  void TriangulationOnCAD::refine_mesh()
  {
    tria.refine_global(1);
  }

  // @sect4{TriangulationOnCAD::output_results}

  // Outputting the results of our computations is a rather mechanical
  // task. All the components of this function have been discussed
  // before:
  void TriangulationOnCAD::output_results(const unsigned int cycle)
  {
    const std::string filename =
      (output_filename + "_" + Utilities::int_to_string(cycle) + ".vtk");
    std::ofstream logfile(filename);
    GridOut       grid_out;
    grid_out.write_vtk(tria, logfile);
  }


  // @sect4{TriangulationOnCAD::run}

  // This is the main function. It should be self explanatory in its
  // briefness:
  void TriangulationOnCAD::run()
  {
    read_domain();

    const unsigned int n_cycles = 5;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        refine_mesh();
        output_results(cycle + 2);
      }
  }
} // namespace Step54


// @sect3{The main() function}

// This is the main function of this program. It is in its basic structure
// like all previous tutorial programs, but runs the main class through the
// three possibilities of new vertex placement:
int main()
{
  try
    {
      using namespace Step54;

      const std::string in_mesh_filename = "../input/HourglassMesh.vtk";
      const std::string cad_file_name    = "../input/HourGlass.IGS";

      std::cout << "----------------------------------------------------------"
                << std::endl;
      std::cout << "Testing projection in direction normal to CAD surface"
                << std::endl;
      std::cout << "----------------------------------------------------------"
                << std::endl;
      std::string        out_mesh_filename = ("3d_mesh_normal_projection");
      TriangulationOnCAD tria_on_cad_norm(in_mesh_filename,
                                          cad_file_name,
                                          out_mesh_filename,
                                          TriangulationOnCAD::NormalProjection);
      tria_on_cad_norm.run();/*
      std::cout << "----------------------------------------------------------"
                << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;

      std::cout << "----------------------------------------------------------"
                << std::endl;
      std::cout << "Testing projection in y-axis direction" << std::endl;
      std::cout << "----------------------------------------------------------"
                << std::endl;
      out_mesh_filename = ("3d_mesh_directional_projection");
      TriangulationOnCAD tria_on_cad_dir(
        in_mesh_filename,
        cad_file_name,
        out_mesh_filename,
        TriangulationOnCAD::DirectionalProjection);
      tria_on_cad_dir.run();
      std::cout << "----------------------------------------------------------"
                << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;

      std::cout << "----------------------------------------------------------"
                << std::endl;
      std::cout << "Testing projection in direction normal to mesh elements"
                << std::endl;
      std::cout << "----------------------------------------------------------"
                << std::endl;
      out_mesh_filename = ("3d_mesh_normal_to_mesh_projection");
      TriangulationOnCAD tria_on_cad_norm_to_mesh(
        in_mesh_filename,
        cad_file_name,
        out_mesh_filename,
        TriangulationOnCAD::NormalToMeshProjection);
      tria_on_cad_norm_to_mesh.run();
      std::cout << "----------------------------------------------------------"
                << std::endl;
      std::cout << std::endl;
      std::cout << std::endl;*/
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }

  return 0;
}
