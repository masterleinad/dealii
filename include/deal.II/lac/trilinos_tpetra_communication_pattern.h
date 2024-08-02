// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2018 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_trilinos_tpetra_communication_pattern_h
#define dealii_trilinos_tpetra_communication_pattern_h


#include <deal.II/base/config.h>

#ifdef DEAL_II_TRILINOS_WITH_TPETRA

#include <deal.II/base/memory_space.h>
#  include <deal.II/base/communication_pattern_base.h>

#  include <Tpetra_Export.hpp>
#  include <Tpetra_Import.hpp>

#  include <memory>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace TpetraWrappers
  {
    /**
     * This class implements a wrapper to Tpetra::Import and Tpetra::Export.
     */
    template <typename MemorySpace = dealii::MemorySpace::Default>
    class CommunicationPattern : public Utilities::MPI::CommunicationPatternBase
    {
    public:
    #  if DEAL_II_TRILINOS_VERSION_GTE(14, 2, 0)
      using NodeType = Tpetra::KokkosCompat::KokkosDeviceWrapperNode<
        typename MemorySpace::kokkos_space::execution_space,
        typename MemorySpace::kokkos_space>;
#  else
      using NodeType = Kokkos::Compat::KokkosDeviceWrapperNode<
        typename MemorySpace::kokkos_space::execution_space,
        typename MemorySpace::kokkos_space>;
#  endif
	  
	    /**
       * Initialize the communication pattern.
       *
       * @param[in] locally_owned_indices The set of indices of elements
       *   in the array mentioned in the class documentation that are
       *   stored on the current process.
       * @param[in] ghost_indices The set of indices of elements in the
       *   array mentioned in the class documentation that the current
       *   process will need to be able to import.
       * @param[in] communicator The MPI communicator used to describe the
       *   entire set of processes that participate in the storage and
       *   access to elements of the array.
       */
      CommunicationPattern(const IndexSet &locally_owned_indices,
                           const IndexSet &ghost_indices,
                           const MPI_Comm  communicator);

      /**
       * Reinitialize the communication pattern.
       *
       * @param[in] locally_owned_indices The set of indices of elements
       *   in the array mentioned in the class documentation that are
       *   stored on the current process.
       * @param[in] ghost_indices The set of indices of elements in the
       *   array mentioned in the class documentation that the current
       *   process will need to be able to import.
       * @param[in] communicator The MPI communicator used to describe the
       *   entire set of processes that participate in the storage and
       *   access to elements of the array.
       */
      virtual void
      reinit(const IndexSet &locally_owned_indices,
             const IndexSet &ghost_indices,
             const MPI_Comm  communicator) override;

      /**
       * Return the underlying MPI communicator.
       */
      virtual MPI_Comm
      get_mpi_communicator() const override;

      /**
       * Return the underlying Tpetra::Import object.
       */
      const Tpetra::Import<int, types::signed_global_dof_index, NodeType> &
      get_tpetra_import() const;

      /**
       * Return a Teuchos::RCP to the underlying Tpetra::Import object.
       */
      Teuchos::RCP<const Tpetra::Import<int, types::signed_global_dof_index, NodeType>>
      get_tpetra_import_rcp() const;

      /**
       * Return the underlying Tpetra::Export object.
       */
      const Tpetra::Export<int, types::signed_global_dof_index, NodeType> &
      get_tpetra_export() const;

      /**
       * Return a Teuchos::RCP to the underlying Tpetra::Export object.
       */
      Teuchos::RCP<const Tpetra::Export<int, types::signed_global_dof_index, NodeType>>
      get_tpetra_export_rcp() const;

    private:
      /**
       * Teuchos::RCP to the MPI communicator used.
       */
      Teuchos::RCP<const MPI_Comm> comm;

      /**
       * Teuchos::RCP to the Tpetra::Import object used.
       */
      Teuchos::RCP<Tpetra::Import<int, types::signed_global_dof_index, NodeType>>
        tpetra_import;

      /**
       * Teuchos::RCP to the Tpetra::Export object used.
       */
      Teuchos::RCP<Tpetra::Export<int, types::signed_global_dof_index, NodeType>>
        tpetra_export;
    };
  } // end of namespace TpetraWrappers
} // end of namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#endif

#endif
