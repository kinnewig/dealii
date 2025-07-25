// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2004 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_petsc_sparse_matrix_h
#define dealii_petsc_sparse_matrix_h


#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_PETSC

#  include <deal.II/lac/exceptions.h>
#  include <deal.II/lac/petsc_matrix_base.h>
#  include <deal.II/lac/petsc_vector.h>

#  include <petscis.h>
#  include <petscistypes.h>

#  include <cstddef>
#  include <vector>

DEAL_II_NAMESPACE_OPEN
// forward declaration
#  ifndef DOXYGEN
template <typename MatrixType>
class BlockMatrixBase;
#  endif

namespace PETScWrappers
{
  /**
   * Implementation of a sequential sparse matrix class based on PETSc. All
   * the functionality is actually in the base class, except for the calls to
   * generate a sequential sparse matrix. This is possible since PETSc only
   * works on an abstract matrix type and internally distributes to functions
   * that do the actual work depending on the actual matrix type (much like
   * using virtual functions). Only the functions creating a matrix of
   * specific type differ, and are implemented in this particular class.
   *
   * @ingroup PETScWrappers
   * @ingroup Matrix1
   */
  class SparseMatrix : public MatrixBase
  {
  public:
    /**
     * A structure that describes some of the traits of this class in terms of
     * its run-time behavior. Some other classes (such as the block matrix
     * classes) that take one or other of the matrix classes as its template
     * parameters can tune their behavior based on the variables in this
     * class.
     */
    struct Traits
    {
      /**
       * It is safe to elide additions of zeros to individual elements of this
       * matrix.
       */
      static const bool zero_addition_can_be_elided = true;
    };

    /**
     * Default constructor. Create an empty matrix.
     */
    SparseMatrix();

    /**
     * Initialize a Matrix from a PETSc Mat object. Note that we do not copy
     * the matrix. The Mat object is referenced by the newly created instance
     * of the class using PetscObjectReference. This is in line with the PETSc
     * approach to object ownership, which mimics std::shared_ptr.
     */
    explicit SparseMatrix(const Mat &);

    /**
     * Create a sparse matrix of dimensions @p m times @p n, with an initial
     * guess of @p n_nonzero_per_row nonzero elements per row. PETSc is able
     * to cope with the situation that more than this number of elements is
     * later allocated for a row, but this involves copying data, and is thus
     * expensive.
     *
     * The @p is_symmetric flag determines whether we should tell PETSc that
     * the matrix is going to be symmetric (as indicated by the call
     * <tt>MatSetOption(mat, MAT_SYMMETRIC)</tt>. Note that the PETSc
     * documentation states that one cannot form an ILU decomposition of a
     * matrix for which this flag has been set to @p true, only an ICC. The
     * default value of this flag is @p false.
     */
    SparseMatrix(const size_type m,
                 const size_type n,
                 const size_type n_nonzero_per_row,
                 const bool      is_symmetric = false);

    /**
     * Initialize a rectangular matrix with @p m rows and @p n columns.  The
     * maximal number of nonzero entries for each row separately is given by
     * the @p row_lengths array.
     *
     * Just as for the other constructors: PETSc is able to cope with the
     * situation that more than this number of elements is later allocated for
     * a row, but this involves copying data, and is thus expensive.
     *
     * The @p is_symmetric flag determines whether we should tell PETSc that
     * the matrix is going to be symmetric (as indicated by the call
     * <tt>MatSetOption(mat, MAT_SYMMETRIC)</tt>. Note that the PETSc
     * documentation states that one cannot form an ILU decomposition of a
     * matrix for which this flag has been set to @p true, only an ICC. The
     * default value of this flag is @p false.
     */
    SparseMatrix(const size_type               m,
                 const size_type               n,
                 const std::vector<size_type> &row_lengths,
                 const bool                    is_symmetric = false);

    /**
     * Initialize a sparse matrix using the given sparsity pattern.
     *
     * Note that PETSc can be very slow if you do not provide it with a good
     * estimate of the lengths of rows. Using the present function is a very
     * efficient way to do this, as it uses the exact number of nonzero
     * entries for each row of the matrix by using the given sparsity pattern
     * argument. If the @p preset_nonzero_locations flag is @p true, this
     * function in addition not only sets the correct row sizes up front, but
     * also pre-allocated the correct nonzero entries in the matrix.
     *
     * PETsc allows to later add additional nonzero entries to a matrix, by
     * simply writing to these elements. However, this will then lead to
     * additional memory allocations which are very inefficient and will
     * greatly slow down your program. It is therefore significantly more
     * efficient to get memory allocation right from the start.
     */
    template <typename SparsityPatternType>
    explicit SparseMatrix(const SparsityPatternType &sparsity_pattern,
                          const bool preset_nonzero_locations = true);

    /**
     * This operator assigns a scalar to a matrix. Since this does usually not
     * make much sense (should we set all matrix entries to this value? Only
     * the nonzero entries of the sparsity pattern?), this operation is only
     * allowed if the actual value to be assigned is zero. This operator only
     * exists to allow for the obvious notation <tt>matrix=0</tt>, which sets
     * all elements of the matrix to zero, but keep the sparsity pattern
     * previously used.
     */
    SparseMatrix &
    operator=(const double d);

    /**
     * The copy constructor is deleted.
     */
    SparseMatrix(const SparseMatrix &) = delete;

    /**
     * The copy assignment operator is deleted.
     */
    SparseMatrix &
    operator=(const SparseMatrix &) = delete;

    /**
     * Throw away the present matrix and generate one that has the same
     * properties as if it were created by the constructor of this class with
     * the same argument list as the present function.
     */
    void
    reinit(const size_type m,
           const size_type n,
           const size_type n_nonzero_per_row,
           const bool      is_symmetric = false);

    /**
     * Throw away the present matrix and generate one that has the same
     * properties as if it were created by the constructor of this class with
     * the same argument list as the present function.
     */
    void
    reinit(const size_type               m,
           const size_type               n,
           const std::vector<size_type> &row_lengths,
           const bool                    is_symmetric = false);

    /**
     * Initialize a sparse matrix using the given sparsity pattern.
     *
     * Note that PETSc can be very slow if you do not provide it with a good
     * estimate of the lengths of rows. Using the present function is a very
     * efficient way to do this, as it uses the exact number of nonzero
     * entries for each row of the matrix by using the given sparsity pattern
     * argument. If the @p preset_nonzero_locations flag is @p true, this
     * function in addition not only sets the correct row sizes up front, but
     * also pre-allocated the correct nonzero entries in the matrix.
     *
     * PETsc allows to later add additional nonzero entries to a matrix, by
     * simply writing to these elements. However, this will then lead to
     * additional memory allocations which are very inefficient and will
     * greatly slow down your program. It is therefore significantly more
     * efficient to get memory allocation right from the start.
     *
     * Despite the fact that it would seem to be an obvious win, setting the
     * @p preset_nonzero_locations flag to @p true doesn't seem to accelerate
     * program. Rather on the contrary, it seems to be able to slow down
     * entire programs somewhat. This is surprising, since we can use
     * efficient function calls into PETSc that allow to create multiple
     * entries at once; nevertheless, given the fact that it is inefficient,
     * the respective flag has a default value equal to @p false.
     */
    template <typename SparsityPatternType>
    void
    reinit(const SparsityPatternType &sparsity_pattern,
           const bool                 preset_nonzero_locations = true);

    /**
     * Return the number of rows of this matrix.
     */
    std::size_t
    m() const;

    /**
     * Return the number of columns of this matrix.
     */
    std::size_t
    n() const;

    /**
     * Perform the matrix-matrix multiplication $C = AB$, or,
     * $C = A \text{diag}(V) B$ given a compatible vector $V$.
     *
     * This function calls MatrixBase::mmult() to do the actual work.
     */
    void
    mmult(SparseMatrix       &C,
          const SparseMatrix &B,
          const MPI::Vector  &V = MPI::Vector()) const;

    /**
     * Perform the matrix-matrix multiplication with the transpose of
     * <tt>this</tt>, i.e., $C = A^T B$, or,
     * $C = A^T \text{diag}(V) B$ given a compatible vector $V$.
     *
     * This function calls MatrixBase::Tmmult() to do the actual work.
     */
    void
    Tmmult(SparseMatrix       &C,
           const SparseMatrix &B,
           const MPI::Vector  &V = MPI::Vector()) const;

  private:
    /**
     * Do the actual work for the respective reinit() function and the
     * matching constructor, i.e. create a matrix.
     */
    void
    do_reinit(const size_type m,
              const size_type n,
              const size_type n_nonzero_per_row,
              const bool      is_symmetric = false);

    /**
     * Same as previous function.
     */
    void
    do_reinit(const size_type               m,
              const size_type               n,
              const std::vector<size_type> &row_lengths,
              const bool                    is_symmetric = false);

    /**
     * Same as previous function.
     */
    template <typename SparsityPatternType>
    void
    do_reinit(const SparsityPatternType &sparsity_pattern,
              const bool                 preset_nonzero_locations);

    // To allow calling protected prepare_add() and prepare_set().
    friend class BlockMatrixBase<SparseMatrix>;
  };

  namespace MPI
  {
    /**
     * Implementation of a parallel sparse matrix class based on PETSc, with
     * rows of the matrix distributed across an MPI network. All the
     * functionality is actually in the base class, except for the calls to
     * generate a parallel sparse matrix. This is possible since PETSc only
     * works on an abstract matrix type and internally distributes to
     * functions that do the actual work depending on the actual matrix type
     * (much like using virtual functions). Only the functions creating a
     * matrix of specific type differ, and are implemented in this particular
     * class.
     *
     * There are a number of comments on the communication model as well as
     * access to individual elements in the documentation to the parallel
     * vector class. These comments apply here as well.
     *
     *
     * <h3>Partitioning of matrices</h3>
     *
     * PETSc partitions parallel matrices so that each MPI process "owns" a
     * certain number of rows (i.e. only this process stores the respective
     * entries in these rows). The number of rows each process owns has to be
     * passed to the constructors and reinit() functions via the argument @p
     * local_rows. The individual values passed as @p local_rows on all the
     * MPI processes of course have to add up to the global number of rows of
     * the matrix.
     *
     * In addition to this, PETSc also partitions the rectangular chunk of the
     * matrix it owns (i.e. the @p local_rows times n() elements in the
     * matrix), so that matrix vector multiplications can be performed
     * efficiently. This column-partitioning therefore has to match the
     * partitioning of the vectors with which the matrix is multiplied, just
     * as the row-partitioning has to match the partitioning of destination
     * vectors. This partitioning is passed to the constructors and reinit()
     * functions through the @p local_columns variable, which again has to add
     * up to the global number of columns in the matrix. The name @p
     * local_columns may be named inappropriately since it does not reflect
     * that only these columns are stored locally, but it reflects the fact
     * that these are the columns for which the elements of incoming vectors
     * are stored locally.
     *
     * To make things even more complicated, PETSc needs a very good estimate
     * of the number of elements to be stored in each row to be efficient.
     * Otherwise it spends most of the time with allocating small chunks of
     * memory, a process that can slow down programs to a crawl if it happens
     * to often. As if a good estimate of the number of entries per row isn't
     * even, it even needs to split this as follows: for each row it owns, it
     * needs an estimate for the number of elements in this row that fall into
     * the columns that are set apart for this process (see above), and the
     * number of elements that are in the rest of the columns.
     *
     * Since in general this information is not readily available, most of the
     * initializing functions of this class assume that all of the number of
     * elements you give as an argument to @p n_nonzero_per_row or by @p
     * row_lengths fall into the columns "owned" by this process, and none
     * into the other ones. This is a fair guess for most of the rows, since
     * in a good domain partitioning, nodes only interact with nodes that are
     * within the same subdomain. It does not hold for nodes on the interfaces
     * of subdomain, however, and for the rows corresponding to these nodes,
     * PETSc will have to allocate additional memory, a costly process.
     *
     * The only way to avoid this is to tell PETSc where the actual entries of
     * the matrix will be. For this, there are constructors and reinit()
     * functions of this class that take a DynamicSparsityPattern object
     * containing all this information. While in the general case it is
     * sufficient if the constructors and reinit() functions know the number
     * of local rows and columns, the functions getting a sparsity pattern
     * also need to know the number of local rows (@p local_rows_per_process)
     * and columns (@p local_columns_per_process) for all other processes, in
     * order to compute which parts of the matrix are which. Thus, it is not
     * sufficient to just count the number of degrees of freedom that belong
     * to a particular process, but you have to have the numbers for all
     * processes available at all processes.
     *
     * @ingroup PETScWrappers
     * @ingroup Matrix1
     */
    class SparseMatrix : public MatrixBase
    {
    public:
      /**
       * Declare type for container size.
       */
      using size_type = types::global_dof_index;

      /**
       * A structure that describes some of the traits of this class in terms
       * of its run-time behavior. Some other classes (such as the block
       * matrix classes) that take one or other of the matrix classes as its
       * template parameters can tune their behavior based on the variables in
       * this class.
       */
      struct Traits
      {
        /**
         * It is not safe to elide additions of zeros to individual elements
         * of this matrix. The reason is that additions to the matrix may
         * trigger @ref GlossCollectiveOperation "collective operations" synchronizing buffers on multiple
         * processes. If an addition is elided on one process, this may lead
         * to other processes hanging in an infinite waiting loop.
         */
        static const bool zero_addition_can_be_elided = false;
      };

      /**
       * Default constructor. Create an empty matrix.
       */
      SparseMatrix();

      /**
       * Initialize a SparseMatrix from a PETSc Mat object. Note that we do not
       * copy the matrix. The Mat object is referenced by the newly created
       * instance of the class using PetscObjectReference. This is in line with
       * the PETSc approach to object ownership, which mimics std::shared_ptr.
       */
      explicit SparseMatrix(const Mat &);

      /**
       * Destructor to free the PETSc object.
       */
      ~SparseMatrix() override;

      /**
       * Initialize using the given sparsity pattern with communication
       * happening over the provided @p communicator.
       *
       * For the meaning of the @p local_rows_per_process and @p
       * local_columns_per_process parameters, see the class documentation.
       *
       * Note that PETSc can be very slow if you do not provide it with a good
       * estimate of the lengths of rows. Using the present function is a very
       * efficient way to do this, as it uses the exact number of nonzero
       * entries for each row of the matrix by using the given sparsity
       * pattern argument. If the @p preset_nonzero_locations flag is @p true,
       * this function in addition not only sets the correct row sizes up
       * front, but also pre-allocated the correct nonzero entries in the
       * matrix.
       *
       * PETsc allows to later add additional nonzero entries to a matrix, by
       * simply writing to these elements. However, this will then lead to
       * additional memory allocations which are very inefficient and will
       * greatly slow down your program. It is therefore significantly more
       * efficient to get memory allocation right from the start.
       */
      template <typename SparsityPatternType>
      SparseMatrix(const MPI_Comm                communicator,
                   const SparsityPatternType    &sparsity_pattern,
                   const std::vector<size_type> &local_rows_per_process,
                   const std::vector<size_type> &local_columns_per_process,
                   const unsigned int            this_process,
                   const bool preset_nonzero_locations = true);

      /**
       * This operator assigns a scalar to a matrix. Since this does usually
       * not make much sense (should we set all matrix entries to this value?
       * Only the nonzero entries of the sparsity pattern?), this operation is
       * only allowed if the actual value to be assigned is zero. This
       * operator only exists to allow for the obvious notation
       * <tt>matrix=0</tt>, which sets all elements of the matrix to zero, but
       * keep the sparsity pattern previously used.
       */
      SparseMatrix &
      operator=(const value_type d);


      /**
       * Make a copy of the PETSc matrix @p other. It is assumed that both
       * matrices have the same SparsityPattern.
       */
      void
      copy_from(const SparseMatrix &other);

      /**
       * Initialize using the given sparsity pattern with communication
       * happening over the provided @p communicator.
       *
       * Note that PETSc can be very slow if you do not provide it with a good
       * estimate of the lengths of rows. Using the present function is a very
       * efficient way to do this, as it uses the exact number of nonzero
       * entries for each row of the matrix by using the given sparsity
       * pattern argument. If the @p preset_nonzero_locations flag is @p true,
       * this function in addition not only sets the correct row sizes up
       * front, but also pre-allocated the correct nonzero entries in the
       * matrix.
       *
       * PETsc allows to later add additional nonzero entries to a matrix, by
       * simply writing to these elements. However, this will then lead to
       * additional memory allocations which are very inefficient and will
       * greatly slow down your program. It is therefore significantly more
       * efficient to get memory allocation right from the start.
       */
      template <typename SparsityPatternType>
      void
      reinit(const MPI_Comm                communicator,
             const SparsityPatternType    &sparsity_pattern,
             const std::vector<size_type> &local_rows_per_process,
             const std::vector<size_type> &local_columns_per_process,
             const unsigned int            this_process,
             const bool                    preset_nonzero_locations = true);

      /**
       * Create a square matrix where the size() of the IndexSet determines the
       * global number of rows and columns and the entries of the IndexSet
       * give the rows and columns for the calling processor. Note that only
       * ascending, 1:1 IndexSets are supported.
       */
      template <typename SparsityPatternType>
      void
      reinit(const IndexSet            &local_partitioning,
             const SparsityPatternType &sparsity_pattern,
             const MPI_Comm             communicator);

      /**
       * Create a matrix where the size() of the IndexSets determine the
       * global number of rows and columns and the entries of the IndexSet
       * give the rows and columns for the calling processor. Note that only
       * ascending, 1:1 IndexSets are supported.
       */
      template <typename SparsityPatternType>
      void
      reinit(const IndexSet            &local_rows,
             const IndexSet            &local_columns,
             const SparsityPatternType &sparsity_pattern,
             const MPI_Comm             communicator);

      /**
       * Initialize this matrix to have the same structure as @p other. This
       * will not copy the values of the other matrix, but you can use
       * copy_from() for this.
       */
      void
      reinit(const SparseMatrix &other);

      /**
       * Create a matrix where the size of the IndexSets determine the
       * global number of rows and columns and the entries of the IndexSet
       * give the rows and columns for the calling processor. Note that only
       * ascending, 1:1 IndexSets are supported. The additional call to the
       * local to global mappings is required to create the matrix of type
       * IS (see DoFTools::extract_locally_active_dofs).
       * This is required by the BDDC preconditioner.
       */
      template <typename SparsityPatternType>
      void
      reinit(const IndexSet            &local_rows,
             const IndexSet            &local_active_rows,
             const IndexSet            &local_columns,
             const IndexSet            &local_active_columns,
             const SparsityPatternType &sparsity_pattern,
             const MPI_Comm             communicator);

      /**
       * @addtogroup Exceptions
       * @{
       */
      /**
       * Exception
       */
      DeclException2(ExcLocalRowsTooLarge,
                     int,
                     int,
                     << "The number of local rows " << arg1
                     << " must be larger than the total number of rows "
                     << arg2);
      /** @} */

      /**
       * Return the square of the norm of the vector $v$ with respect to the
       * norm induced by this matrix, i.e. $\left(v^\ast,Mv\right)$. This is
       * useful, e.g. in the finite element context, where the $L_2$ norm of a
       * function equals the matrix norm with respect to the @ref GlossMassMatrix "mass matrix" of
       * the vector representing the nodal values of the finite element
       * function.
       *
       * Obviously, the matrix needs to be quadratic for this operation.
       *
       * The implementation of this function is not as efficient as the one in
       * the @p MatrixBase class used in deal.II (i.e. the original one, not
       * the PETSc wrapper class) since PETSc doesn't support this operation
       * and needs a temporary vector.
       */
      PetscScalar
      matrix_norm_square(const Vector &v) const;

      /**
       * Compute the matrix scalar product $\left(u^\ast,Mv\right)$.
       *
       * The implementation of this function is not as efficient as the one in
       * the @p MatrixBase class used in deal.II (i.e. the original one, not
       * the PETSc wrapper class) since PETSc doesn't support this operation
       * and needs a temporary vector.
       */
      PetscScalar
      matrix_scalar_product(const Vector &u, const Vector &v) const;

      /**
       * Return the partitioning of the domain space of this matrix, i.e., the
       * partitioning of the vectors this matrix has to be multiplied with.
       */
      IndexSet
      locally_owned_domain_indices() const;

      /**
       * Return the partitioning of the range space of this matrix, i.e., the
       * partitioning of the vectors that result from matrix-vector
       * products.
       */
      IndexSet
      locally_owned_range_indices() const;

      /**
       * Perform the matrix-matrix multiplication $C = AB$, or,
       * $C = A \text{diag}(V) B$ given a compatible vector $V$.
       *
       * This function calls MatrixBase::mmult() to do the actual work.
       */
      void
      mmult(SparseMatrix       &C,
            const SparseMatrix &B,
            const MPI::Vector  &V = MPI::Vector()) const;

      /**
       * Perform the matrix-matrix multiplication with the transpose of
       * <tt>this</tt>, i.e., $C = A^T B$, or,
       * $C = A^T \text{diag}(V) B$ given a compatible vector $V$.
       *
       * This function calls MatrixBase::Tmmult() to do the actual work.
       */
      void
      Tmmult(SparseMatrix       &C,
             const SparseMatrix &B,
             const MPI::Vector  &V = MPI::Vector()) const;

    private:
      /**
       * Same as previous functions.
       */
      template <typename SparsityPatternType>
      void
      do_reinit(const MPI_Comm                comm,
                const SparsityPatternType    &sparsity_pattern,
                const std::vector<size_type> &local_rows_per_process,
                const std::vector<size_type> &local_columns_per_process,
                const unsigned int            this_process,
                const bool                    preset_nonzero_locations);

      /**
       * Same as previous functions.
       */
      template <typename SparsityPatternType>
      void
      do_reinit(const MPI_Comm             comm,
                const IndexSet            &local_rows,
                const IndexSet            &local_columns,
                const SparsityPatternType &sparsity_pattern);

      /**
       * Same as previous functions, but here we consider active dofs for
       * matrices of IS type.
       */
      template <typename SparsityPatternType>
      void
      do_reinit(const MPI_Comm             comm,
                const IndexSet            &local_rows,
                const IndexSet            &local_active_rows,
                const IndexSet            &local_columns,
                const IndexSet            &local_active_columns,
                const SparsityPatternType &sparsity_pattern);

      // To allow calling protected prepare_add() and prepare_set().
      friend class BlockMatrixBase<SparseMatrix>;
    };

  } // namespace MPI
} // namespace PETScWrappers

DEAL_II_NAMESPACE_CLOSE

#else

// Make sure the scripts that create the C++20 module input files have
// something to latch on if the preprocessor #ifdef above would
// otherwise lead to an empty content of the file.
DEAL_II_NAMESPACE_OPEN
DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_PETSC

#endif
