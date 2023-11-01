// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2023 by the deal.II authors
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

#ifndef dealii_trilinos_tpetra_sparse_matrix_templates_h
#define dealii_trilinos_tpetra_sparse_matrix_templates_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_TRILINOS_WITH_TPETRA

#  include <deal.II/lac/dynamic_sparsity_pattern.h>
#  include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#  include <deal.II/lac/trilinos_tpetra_sparsity_pattern.h>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{

  namespace TpetraWrappers
  {

    namespace
    {
      using size_type = dealii::types::signed_global_dof_index;

      using MapType = Tpetra::Map<int, dealii::types::signed_global_dof_index>;

      template <typename Number>
      using MatrixType =
        Tpetra::CrsMatrix<Number, int, dealii::types::signed_global_dof_index>;

      using GraphType =
        Tpetra::CrsGraph<int, dealii::types::signed_global_dof_index>;

      using ExportType =
        Tpetra::Export<int, dealii::types::signed_global_dof_index>;


      template <typename Number, typename SparsityPatternType>
      void
      reinit_matrix(const IndexSet            &row_parallel_partitioning,
                    const IndexSet            &column_parallel_partitioning,
                    const SparsityPatternType &sparsity_pattern,
                    const bool                 exchange_data,
                    const MPI_Comm             communicator,
                    Teuchos::RCP<MapType>     &column_space_map,
                    Teuchos::RCP<MatrixType<Number>> &matrix)
      {
        // release memory before reallocation
        matrix.reset();

        // Get the Tpetra::Maps
        Teuchos::RCP<MapType> row_space_map =
          row_parallel_partitioning.make_tpetra_map_rcp(communicator, false);

        column_space_map =
          column_parallel_partitioning.make_tpetra_map_rcp(communicator, false);

        // if we want to exchange data, build a usual Trilinos sparsity pattern
        // and let that handle the exchange. otherwise, manually create a
        // CrsGraph, which consumes considerably less memory because it can set
        // correct number of indices right from the start
        if (exchange_data)
        {
          SparsityPattern trilinos_sparsity;
          trilinos_sparsity.reinit(row_parallel_partitioning,
                                   column_parallel_partitioning,
                                   sparsity_pattern,
                                   communicator,
                                   exchange_data);
          matrix = Teuchos::rcp(new MatrixType<Number>(
            trilinos_sparsity.trilinos_sparsity_pattern()));

          return;
        }

        IndexSet relevant_rows(sparsity_pattern.row_index_set());
        // serial case
        if (relevant_rows.size() == 0)
        {
          relevant_rows.set_size(row_space_map->getGlobalNumElements());
          relevant_rows.add_range(0, row_space_map->getGlobalNumElements());
        }
        relevant_rows.compress();


        const unsigned int n_rows = relevant_rows.n_elements();
        std::vector<TrilinosWrappers::types::int_type> ghost_rows;
        std::vector<unsigned long int> n_entries_per_row(row_space_map->getLocalNumElements());
        //std::vector<unsigned long int> n_entries_per_ghost_row;
        for (unsigned int i = 0, own = 0; i < n_rows; ++i)
        {
          const TrilinosWrappers::types::int_type global_row = relevant_rows.nth_index_in_set(i);
          if (row_space_map->isNodeGlobalElement(global_row))
            n_entries_per_row[own++] = sparsity_pattern.row_length(global_row);
        }

        // Construct the Teuchos::CrsGraph
        Teuchos::RCP<GraphType> graph;

        Teuchos::ArrayView<const unsigned long int> array(n_entries_per_row);
        // parallel
        if (row_space_map->getComm()->getSize() > 1)
          {
            graph =
              Teuchos::rcp(new GraphType(row_space_map, array));
          }
        // sequentiel
        else
          {
            graph = Teuchos::rcp(new GraphType(row_space_map,
                                               column_space_map,
                                               array));
          }

        // now insert the indices, select between the right matrix
        std::vector<TrilinosWrappers::types::int_type> row_indices;

        for (unsigned int i = 0; i < n_rows; ++i)
          {
            const TrilinosWrappers::types::int_type global_row = relevant_rows.nth_index_in_set(i);
            const int row_length = sparsity_pattern.row_length(global_row);
            if (row_length == 0)
              continue;

            row_indices.resize(row_length, -1);
            for (int col = 0; col < row_length; ++col)
              row_indices[col] = sparsity_pattern.column_number(global_row, col);

            if (row_space_map->isNodeGlobalElement(global_row))
              graph->insertGlobalIndices(global_row,
                                         row_length,
                                         row_indices.data());
          }

        // Eventually, optimize the graph structure (sort indices, make memory
        // contiguous, etc). note that the documentation of the function indeed
        // states that we first need to provide the column (domain) map and then
        // the row (range) map
        graph->fillComplete(column_space_map, row_space_map);

        // And now finally generate the matrix.
        matrix = Teuchos::rcp(new MatrixType<Number>(graph));
      }



      template void
      reinit_matrix(const IndexSet               &row_parallel_partitioning,
                    const IndexSet               &column_parallel_partitioning,
                    const DynamicSparsityPattern &sparsity_pattern,
                    const bool                    exchange_data,
                    const MPI_Comm                communicator,
                    Teuchos::RCP<MapType>        &column_space_map,
                    Teuchos::RCP<MatrixType<double>> &matrix);
    } // namespace



    // The constructor is actually the
    // only point where we have to check
    // whether we build a serial or a
    // parallel Trilinos matrix.
    // Actually, it does not even matter
    // how many threads there are, but
    // only if we use an MPI compiler or
    // a standard compiler. So, even one
    // thread on a configuration with
    // MPI will still get a parallel
    // interface.
    template <typename Number>
    SparseMatrix<Number>::SparseMatrix()
      : column_space_map(
          new MapType(0, 0, Utilities::Trilinos::tpetra_comm_self()))
      , matrix()
      , last_action(Tpetra::ZERO)
      , fill_complete(true)
    {
      // Prepare the graph
      Teuchos::RCP<GraphType> graph(
        new GraphType(column_space_map, column_space_map, 0));
      graph->fillComplete();

      // Create the matrix from the graph
      matrix = Teuchos::rcp(new MatrixType(graph));
    }



    template <typename Number>
    template <typename SparsityPatternType>
    void
    SparseMatrix<Number>::reinit(const IndexSet &row_parallel_partitioning,
                                 const IndexSet &col_parallel_partitioning,
                                 const SparsityPatternType &sparsity_pattern,
                                 const MPI_Comm             communicator,
                                 const bool                 exchange_data)
    {
      reinit_matrix(row_parallel_partitioning,
                    col_parallel_partitioning,
                    sparsity_pattern,
                    exchange_data,
                    communicator,
                    column_space_map,
                    matrix);

      last_action   = Tpetra::ZERO;
      fill_complete = true;
    }



    template <typename Number>
    void
    SparseMatrix<Number>::compress(VectorOperation::values operation)
    {
      //TODO
      matrix->fillComplete(column_space_map, matrix->getRowMap());

      last_action   = Tpetra::ZERO;
      fill_complete = true;
    }



    template <typename Number>
    SparseMatrix<Number> &
    SparseMatrix<Number>::operator=(const double d)
    {
      // TODO: Assert
      // Assert(d == 0, ExcScalarAssignmentOnlyForTpetra::ZEROValue());

      if (fill_complete)
        {
          matrix->resumeFill();
          fill_complete = false;
        }

      matrix->setAllToScalar(d);

      return *this;
    }


    template <typename Number>
    void
    SparseMatrix<Number>::add(const size_type       row,
                              const size_type       n_cols,
                              const size_type      *col_indices,
                              const TrilinosScalar *values,
                              const bool            elide_zero_values,
                              const bool /*col_indices_are_sorted*/)
    {
      AssertIndexRange(row, this->m());

      if (fill_complete)
        {
          matrix->resumeFill();
          fill_complete = false;
        }

      const TrilinosWrappers::types::int_type *col_index_ptr;
      const TrilinosScalar                    *col_value_ptr;
      TrilinosWrappers::types::int_type        n_columns;

      boost::container::small_vector<TrilinosScalar, 100> local_value_array(
        n_cols);
      boost::container::small_vector<TrilinosWrappers::types::int_type, 100>
        local_index_array(n_cols);

      // If we don't elide zeros, the pointers are already available... need to
      // cast to non-const pointers as that is the format taken by Trilinos (but
      // we will not modify const data)
      if (elide_zero_values == false)
        {
          col_index_ptr =
            reinterpret_cast<const TrilinosWrappers::types::int_type *>(
              col_indices);
          col_value_ptr = values;
          n_columns     = n_cols;
#  ifdef DEBUG
          for (size_type j = 0; j < n_cols; ++j)
            AssertIsFinite(values[j]);
#  endif
        }
      else
        {
          // Otherwise, extract nonzero values in each row and the corresponding
          // index.
          col_index_ptr = local_index_array.data();
          col_value_ptr = local_value_array.data();

          n_columns = 0;
          for (size_type j = 0; j < n_cols; ++j)
            {
              const double value = values[j];

              AssertIsFinite(value);
              if (value != 0)
                {
                  local_index_array[n_columns] = col_indices[j];
                  local_value_array[n_columns] = value;
                  ++n_columns;
                }
            }

          AssertIndexRange(n_columns, n_cols + 1);
        }
      // Exit early if there is nothing to do
      if (n_columns == 0)
        {
          return;
        }

      // If the calling processor owns the row to which we want to add values,
      // we can directly call the Tpetra::CrsMatrix<double, int,
      // types::signed_global_dof_index> input function, which is much faster
      // than the Tpetra::FECrsMatrix<double, int,
      // dealii::types::signed_global_dof_index> function.
      if (matrix->getRowMap()->isNodeGlobalElement(
            static_cast<TrilinosWrappers::types::int_type>(row)) == true)
        {
          matrix->MatrixType::sumIntoGlobalValues(row,
                                                  n_columns,
                                                  col_value_ptr,
                                                  col_index_ptr);
        }
      else
        {
          // When we're at off-processor data, we have to stick with the
          // standard SumIntoGlobalValues function. Nevertheless, the way we
          // call it is the fastest one (any other will lead to repeated
          // allocation and deallocation of memory in order to call the function
          // we already use, which is very inefficient if writing one element at
          // a time).
          matrix->sumIntoGlobalValues(row,
                                      n_columns,
                                      col_value_ptr,
                                      col_index_ptr);
        }
    }

  } // namespace TpetraWrappers

} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#endif // DEALII_TRILINOS_TPETRA_SPARSE_MATRIX_TEMPLATES_H

#endif // DEAL_II_TRILINOS_WITH_TPETRA
