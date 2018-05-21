// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2018 by the deal.II authors
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

#ifndef dealii_sparsity_tools_h
#define dealii_sparsity_tools_h

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <memory>
#include <vector>

#ifdef DEAL_II_WITH_MPI
#  include <deal.II/base/index_set.h>
#  include <mpi.h>
#endif

DEAL_II_NAMESPACE_OPEN

/*! @addtogroup Sparsity
 *@{
 */

/**
 * A namespace for functions that deal with things that one can do on sparsity
 * patterns, such as renumbering rows and columns (or degrees of freedom if
 * you want) according to the connectivity, or partitioning degrees of
 * freedom.
 */
namespace SparsityTools
{
  /**
   * Enumerator with options for partitioner
   */
  enum class Partitioner
  {
    /**
     * Use METIS partitioner.
     */
    metis = 0,
    /**
     * Use ZOLTAN partitioner.
     */
    zoltan
  };

  /**
   * Use a partitioning algorithm to generate a partitioning of the degrees of
   * freedom represented by this sparsity pattern. In effect, we view this
   * sparsity pattern as a graph of connections between various degrees of
   * freedom, where each nonzero entry in the sparsity pattern corresponds to
   * an edge between two nodes in the connection graph. The goal is then to
   * decompose this graph into groups of nodes so that a minimal number of
   * edges are cut by the boundaries between node groups. This partitioning is
   * done by METIS or ZOLTAN, depending upon which partitioner is chosen
   * in the fourth argument. The default is METIS. Note that METIS and
   * ZOLTAN can only partition symmetric sparsity patterns, and that of
   * course the sparsity pattern has to be square. We do not check for
   * symmetry of the sparsity pattern, since this is an expensive operation,
   * but rather leave this as the responsibility of caller of this function.
   *
   * After calling this function, the output array will have values between
   * zero and @p n_partitions-1 for each node (i.e. row or column of the
   * matrix).
   *
   * If deal.II was not installed with packages ZOLTAN or METIS, this
   * function will generate an error when corresponding partition method
   * is chosen, unless @p n_partitions is one. I.e., you can write a program
   * so that it runs in the single-processor single-partition case without
   * the packages installed, and only requires them installed when
   * multiple partitions are required.
   *
   * Note that the sparsity pattern itself is not changed by calling this
   * function. However, you will likely use the information generated by
   * calling this function to renumber degrees of freedom, after which you
   * will of course have to regenerate the sparsity pattern.
   *
   * This function will rarely be called separately, since in finite element
   * methods you will want to partition the mesh, not the matrix. This can be
   * done by calling @p GridTools::partition_triangulation.
   */
  void
  partition(const SparsityPattern&     sparsity_pattern,
            const unsigned int         n_partitions,
            std::vector<unsigned int>& partition_indices,
            const Partitioner          partitioner = Partitioner::metis);


  /**
   * This function performs the same operation as the one above, except that
   * it takes into consideration a set of @p cell_weights, which allow the
   * partitioner to balance the graph while taking into consideration the
   * computational effort expended on each cell.
   *
   * @note If the @p cell_weights vector is empty, then no weighting is taken
   * into consideration. If not then the size of this vector must equal to the
   * number of active cells in the triangulation.
   */
  void
  partition(const SparsityPattern&           sparsity_pattern,
            const std::vector<unsigned int>& cell_weights,
            const unsigned int               n_partitions,
            std::vector<unsigned int>&       partition_indices,
            const Partitioner                partitioner = Partitioner::metis);

  /**
   * Using a coloring algorithm provided by ZOLTAN to color nodes whose
   * connections are represented using a SparsityPattern object. In effect,
   * we view this sparsity pattern as a graph of connections between nodes,
   * where each nonzero entry in the @p sparsity_pattern corresponds to
   * an edge between two nodes. The goal is to assign each node a color index
   * such that no two directly connected nodes have the same color.
   * The assigned colors are listed in @p color_indices indexed from one and
   * the function returns total number of colors used. ZOLTAN coloring
   * algorithm is run in serial by each processor. Hence all processors have
   * coloring information of all the nodes. A wrapper function to this
   * function is available in GraphColoring namespace as well.
   *
   * Note that this function requires that SparsityPattern be symmetric
   * (and hence square) i.e an undirected graph representation. We do not
   * check for symmetry of the sparsity pattern, since this is an expensive
   * operation, but rather leave this as the responsibility of caller of
   * this function.
   *
   * @image html color_undirected_graph.png
   * The usage of the function is illustrated by the image above, showing five
   * nodes numbered from 0 to 4. The connections shown are bidirectional.
   * After coloring, it is clear that no two directly connected nodes are
   * assigned the same color.
   *
   * If deal.II was not installed with package ZOLTAN, this function will
   * generate an error.
   *
   * @note The current function is an alternative to
   * GraphColoring::make_graph_coloring() which is tailored to graph
   * coloring arising in shared-memory parallel assembly of matrices.
   */
  unsigned int
  color_sparsity_pattern(const SparsityPattern&     sparsity_pattern,
                         std::vector<unsigned int>& color_indices);

  /**
   * For a given sparsity pattern, compute a re-enumeration of row/column
   * indices based on the algorithm by Cuthill-McKee.
   *
   * This algorithm is a graph renumbering algorithm in which we attempt to
   * find a new numbering of all nodes of a graph based on their connectivity
   * to other nodes (i.e. the edges that connect nodes). This connectivity is
   * here represented by the sparsity pattern. In many cases within the
   * library, the nodes represent degrees of freedom and edges are nonzero
   * entries in a matrix, i.e. pairs of degrees of freedom that couple through
   * the action of a bilinear form.
   *
   * The algorithms starts at a node, searches the other nodes for those which
   * are coupled with the one we started with and numbers these in a certain
   * way. It then finds the second level of nodes, namely those that couple
   * with those of the previous level (which were those that coupled with the
   * initial node) and numbers these. And so on. For the details of the
   * algorithm, especially the numbering within each level, we refer the
   * reader to the book of Schwarz (H. R. Schwarz: Methode der finiten
   * Elemente).
   *
   * These algorithms have one major drawback: they require a good starting
   * node, i.e. node that will have number zero in the output array. A
   * starting node forming the initial level of nodes can thus be given by the
   * user, e.g. by exploiting knowledge of the actual topology of the domain.
   * It is also possible to give several starting indices, which may be used
   * to simulate a simple upstream numbering (by giving the inflow nodes as
   * starting values) or to make preconditioning faster (by letting the
   * Dirichlet boundary indices be starting points).
   *
   * If no starting index is given, one is chosen automatically, namely one
   * with the smallest coordination number (the coordination number is the
   * number of other nodes this node couples with). This node is usually
   * located on the boundary of the domain. There is, however, large ambiguity
   * in this when using the hierarchical meshes used in this library, since in
   * most cases the computational domain is not approximated by tilting and
   * deforming elements and by plugging together variable numbers of elements
   * at vertices, but rather by hierarchical refinement. There is therefore a
   * large number of nodes with equal coordination numbers. The renumbering
   * algorithms will therefore not give optimal results.
   *
   * If the graph has two or more unconnected components and if no starting
   * indices are given, the algorithm will number each component
   * consecutively. However, this requires the determination of a starting
   * index for each component; as a consequence, the algorithm will produce an
   * exception if starting indices are given, taking the latter as an
   * indication that the caller of the function would like to override the
   * part of the algorithm that chooses starting indices.
   */
  void
  reorder_Cuthill_McKee(
    const DynamicSparsityPattern&                         sparsity,
    std::vector<DynamicSparsityPattern::size_type>&       new_indices,
    const std::vector<DynamicSparsityPattern::size_type>& starting_indices
    = std::vector<DynamicSparsityPattern::size_type>());

  /**
   * For a given sparsity pattern, compute a re-enumeration of row/column
   * indices in a hierarchical way, similar to what
   * DoFRenumbering::hierarchical does for degrees of freedom on
   * hierarchically refined meshes.
   *
   * This algorithm first selects a node with the minimum number of neighbors
   * and puts that node and its direct neighbors into one chunk. Next, it
   * selects one of the neighbors of the already selected nodes, adds the node
   * and its direct neighbors that are not part of one of the previous chunks,
   * into the next. After this sweep, neighboring nodes are grouped together.
   * To ensure a similar grouping on a more global level, this grouping is
   * called recursively on the groups so formed. The recursion stops when no
   * further grouping is possible. Eventually, the ordering obtained by this
   * method passes through the indices represented in the sparsity pattern in
   * a z-like way.
   *
   * If the graph has two or more unconnected components, the algorithm will
   * number each component consecutively, starting with the components with
   * the lowest number of nodes.
   */
  void
  reorder_hierarchical(
    const DynamicSparsityPattern&                   sparsity,
    std::vector<DynamicSparsityPattern::size_type>& new_indices);

#ifdef DEAL_II_WITH_MPI
  /**
   * Communicate rows in a dynamic sparsity pattern over MPI.
   *
   * @param dsp A dynamic sparsity pattern that has been built locally and for
   * which we need to exchange entries with other processors to make sure that
   * each processor knows all the elements of the rows of a matrix it stores
   * and that may eventually be written to. This sparsity pattern will be
   * changed as a result of this function: All entries in rows that belong to
   * a different processor are sent to them and added there.
   *
   * @param rows_per_cpu A vector containing the number of of rows per CPU for
   * determining ownership. This is typically the value returned by
   * DoFHandler::n_locally_owned_dofs_per_processor.
   *
   * @param mpi_comm The MPI communicator shared between the processors that
   * participate in this operation.
   *
   * @param myrange The range of elements stored locally. This should be the
   * one used in the constructor of the DynamicSparsityPattern, and should
   * also be the locally relevant set. Only rows contained in myrange are
   * checked in dsp for transfer. This function needs to be used with
   * PETScWrappers::MPI::SparseMatrix for it to work correctly in a parallel
   * computation.
   */
  void
  distribute_sparsity_pattern(
    DynamicSparsityPattern&                               dsp,
    const std::vector<DynamicSparsityPattern::size_type>& rows_per_cpu,
    const MPI_Comm&                                       mpi_comm,
    const IndexSet&                                       myrange);

  /**
   * Similar to the function above, but for BlockDynamicSparsityPattern
   * instead.
   *
   * @param[in,out] dsp The locally built sparsity pattern to be modified.
   * @param owned_set_per_cpu Typically the value given by
   * DoFHandler::locally_owned_dofs_per_processor.
   *
   * @param mpi_comm The MPI communicator to use.
   *
   * @param myrange Typically the locally relevant DoFs.
   */
  void
  distribute_sparsity_pattern(BlockDynamicSparsityPattern& dsp,
                              const std::vector<IndexSet>& owned_set_per_cpu,
                              const MPI_Comm&              mpi_comm,
                              const IndexSet&              myrange);

#endif

  /**
   * Exception
   */
  DeclExceptionMsg(ExcMETISNotInstalled,
                   "The function you called requires METIS, but you did not "
                   "configure deal.II with METIS.");

  /**
   * Exception
   */
  DeclException1(ExcInvalidNumberOfPartitions,
                 int,
                 << "The number of partitions you gave is " << arg1
                 << ", but must be greater than zero.");

  /**
   * Exception
   */
  DeclException1(ExcMETISError,
                 int,
                 << "    An error with error number " << arg1
                 << " occurred while calling a METIS function");

  /**
   * Exception
   */
  DeclException2(ExcInvalidArraySize,
                 int,
                 int,
                 << "The array has size " << arg1 << " but should have size "
                 << arg2);
  /**
   * Exception
   */
  DeclExceptionMsg(
    ExcZOLTANNotInstalled,
    "The function you called requires ZOLTAN, but you did not "
    "configure deal.II with ZOLTAN or zoltan_cpp.h is not available.");
} // namespace SparsityTools

/**
 * @}
 */

DEAL_II_NAMESPACE_CLOSE

#endif
