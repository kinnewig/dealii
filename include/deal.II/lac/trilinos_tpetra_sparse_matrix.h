// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2023 by the deal.II authors
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

#ifndef dealii_trilinos_tpetra_sparse_matrix_h
#define dealii_trilinos_tpetra_sparse_matrix_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_TRILINOS_WITH_TPETRA

#  include <deal.II/base/index_set.h>
#  include <deal.II/base/subscriptor.h>
#  include <deal.II/base/trilinos_utilities.h>

#  include <deal.II/lac/trilinos_tpetra_sparsity_pattern.h>
#  include <deal.II/lac/trilinos_tpetra_vector.h>

// Tpetra includes
#  include <Tpetra_Core.hpp>
#  include <Tpetra_CrsMatrix.hpp>


DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{

  namespace TpetraWrappers
  {

    template <typename Number>
    class SparseMatrix : public Subscriptor
    {
    public:
      // Declerations:
      /**
       * Declare the type for container size.
       */
      using size_type = dealii::types::global_dof_index;

      /**
       * Declare an alias in analogy to all the other container classes.
       */
      using value_type = Number;

      /**
       * Typedef for Tpetra::CrsMatrix
       */
      using MatrixType =
        Tpetra::CrsMatrix<Number, int, dealii::types::signed_global_dof_index>;

      /**
       * Typedef for Tpetra::Map
       */
      using MapType = Tpetra::Map<int, dealii::types::signed_global_dof_index>;

      /**
       * Typedef for Tpetra::CrsGraph
       */
      using GraphType =
        Tpetra::CrsGraph<int, dealii::types::signed_global_dof_index>;


      // Constructor
      /**
       * @name Constructors and initialization.
       */
      /** @{ */
      /**
       * Default constructor. Generates an empty (zero-size) matrix.
       */
      explicit SparseMatrix();

      /**
       * Destructor. Made virtual so that one can use pointers to this class.
       */
      virtual ~SparseMatrix() override = default;


      // Reinit
      /**
       * This function is similar to the other initialization function above,
       * but now also reassigns the matrix rows and columns according to two
       * user-supplied index sets.  To be used for rectangular matrices. The
       * optional argument @p exchange_data can be used for reinitialization
       * with a sparsity pattern that is not fully constructed. This feature is
       * only implemented for input sparsity patterns of type
       * DynamicSparsityPattern.
       *
       * This is a collective operation that needs to be called on all
       * processors in order to avoid a dead lock.
       */
      template <typename SparsityPatternType>
      void
      reinit(const IndexSet            &row_parallel_partitioning,
             const IndexSet            &col_parallel_partitioning,
             const SparsityPatternType &sparsity_pattern,
             const MPI_Comm             communicator  = MPI_COMM_WORLD,
             const bool                 exchange_data = false);


      /** @} */


      /**
       * @name Information on the matrix
       */
      /** @{ */

      /**
       * Return the number of rows in this matrix.
       */
      inline dealii::types::signed_global_dof_index
      m() const;

      /**
       * Return the number of columns in this matrix.
       */
      inline dealii::types::signed_global_dof_index
      n() const;


      /** @} */


      /**
       * @name Modifying entries
       */
      /** @{ */

      /**
       * This operator assigns a scalar to a matrix. Since this does usually not
       * make much sense (should we set all matrix entries to this value?  Only
       * the nonzero entries of the sparsity pattern?), this operation is only
       * allowed if the actual value to be assigned is zero. This operator only
       * exists to allow for the obvious notation <tt>matrix=0</tt>, which sets
       * all elements of the matrix to zero, but keeps the sparsity pattern
       * previously used.
       */
      SparseMatrix &
      operator=(const double d);



      /**
       * Add @p value to the element (<i>i,j</i>).
       *
       * Just as the respective call in deal.II SparseMatrix<Number> class (but
       * in contrast to the situation for PETSc based matrices), this function
       * throws an exception if an entry does not exist in the sparsity pattern.
       * Moreover, if <tt>value</tt> is not a finite number an exception is
       * thrown.
       */
      void
      add(const size_type i, const size_type j, const TrilinosScalar value);

      /**
       * Add an array of values given by <tt>values</tt> in the given global
       * matrix row at columns specified by col_indices in the sparse matrix.
       *
       * Just as the respective call in deal.II SparseMatrix<Number> class (but
       * in contrast to the situation for PETSc based matrices), this function
       * throws an exception if an entry does not exist in the sparsity pattern.
       *
       * The optional parameter <tt>elide_zero_values</tt> can be used to
       * specify whether zero values should be added anyway or these should be
       * filtered away and only non-zero data is added. The default value is
       * <tt>true</tt>, i.e., zero values won't be added into the matrix.
       */
      void
      add(const size_type       row,
          const size_type       n_cols,
          const size_type      *col_indices,
          const TrilinosScalar *values,
          const bool            elide_zero_values      = true,
          const bool            col_indices_are_sorted = false);

      /** @} */

      /**
       * @name Mixed Stuff
       */
      /** @{ */
      /**
       * This command does two things:
       * <ul>
       * <li> If the matrix was initialized without a sparsity pattern, elements
       * have been added manually using the set() command. When this process is
       * completed, a call to compress() reorganizes the internal data
       * structures (sparsity pattern) so that a fast access to data is possible
       * in matrix-vector products.
       * <li> If the matrix structure has already been fixed (either by
       * initialization with a sparsity pattern or by calling compress() during
       * the setup phase), this command does the %parallel exchange of data.
       * This is necessary when we perform assembly on more than one (MPI)
       * process, because then some non-local row data will accumulate on nodes
       * that belong to the current's processor element, but are actually held
       * by another. This command is usually called after all elements have been
       * traversed.
       * </ul>
       *
       * In both cases, this function compresses the data structures and allows
       * the resulting matrix to be used in all other operations like matrix-
       * vector products. This is a collective operation, i.e., it needs to be
       * run on all processors when used in %parallel.
       *
       * See
       * @ref GlossCompress "Compressing distributed objects"
       * for more information.
       */
      void
      compress(VectorOperation::values operation);

      const MatrixType &
      trilinos_matrix() const;

      Teuchos::RCP<MatrixType>
      trilinos_rcp() const;

      /** @} */

    private:
      /**
       * Pointer to the user-supplied Tpetra Trilinos mapping of the matrix
       * columns that assigns parts of the matrix to the individual processes.
       */
      Teuchos::RCP<MapType> column_space_map;

      /**
       * A sparse matrix object in Trilinos to be used for finite element based
       * problems which allows for assembling into non-local elements.  The
       * actual type, a sparse matrix, is set in the constructor.
       */
      Teuchos::RCP<MatrixType> matrix;

      /**
       * Trilinos doesn't allow to mix additions to matrix entries and
       * overwriting them (to make synchronization of %parallel computations
       * simpler). The way we do it is to, for each access operation, store
       * whether it is an insertion or an addition. If the previous one was of
       * different type, then we first have to flush the Trilinos buffers;
       * otherwise, we can simply go on. Luckily, Trilinos has an object for
       * this which does already all the %parallel communications in such a
       * case, so we simply use their model, which stores whether the last
       * operation was an addition or an insertion.
       */
      Tpetra::CombineMode last_action;

      /**
       * A boolean variable to hold information on whether the matrix is
       * fill complete or if the matrix is in compute mode
       */
      bool fill_complete;

    }; // class SparseMatrix



    template <typename Number>
    inline void
    SparseMatrix<Number>::add(const size_type      i,
                              const size_type      j,
                              const TrilinosScalar value)
    {
      AssertIsFinite(value);

      if (value == 0)
        {
          // we have to check after Insert/Add in any case to be consistent
          // with the MPI communication model, but we can save some
          // work if the addend is zero. However, these actions are done in case
          // we pass on to the other function.

          // TODO: fix this (do not run compress here, but fail)
          if (last_action == Tpetra::INSERT)
            matrix->globalAssemble();

          last_action = Tpetra::ADD;

          return;
        }
      else
        add(i, 1, &j, &value, false);
    }



    template <typename Number>
    inline dealii::types::signed_global_dof_index
    SparseMatrix<Number>::m() const
    {
      return matrix->getRowMap()->getGlobalNumElements();
    }



    template <typename Number>
    inline dealii::types::signed_global_dof_index
    SparseMatrix<Number>::n() const
    {
      // If the matrix structure has not been fixed (i.e., we did not have a
      // sparsity pattern), it does not know about the number of columns so we
      // must always take this from the additional column space map
      Assert(column_space_map.get() != nullptr, ExcInternalError());
      return column_space_map->getGlobalNumElements();
    }



    template <typename Number>
    inline const Tpetra::
      CrsMatrix<Number, int, types::signed_global_dof_index> &
      SparseMatrix<Number>::trilinos_matrix() const
    {
      return *matrix;
    }



    template <typename Number>
    inline Teuchos::RCP<
      Tpetra::CrsMatrix<Number, int, types::signed_global_dof_index>>
    SparseMatrix<Number>::trilinos_rcp() const
    {
      return matrix;
    }

  } // namespace TpetraWrappers

} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_TPETRA

#endif // dealii_trilinos_tpetra_sparse_matrix_h
