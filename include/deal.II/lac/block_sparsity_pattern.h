// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2000 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_block_sparsity_pattern_h
#define dealii_block_sparsity_pattern_h


#include <deal.II/base/config.h>

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/observer_pointer.h>
#include <deal.II/base/table.h>

#include <deal.II/lac/block_indices.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern_base.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
template <typename number>
class BlockSparseMatrix;
class BlockDynamicSparsityPattern;
#endif

/**
 * @addtogroup Sparsity
 * @{
 */


/**
 * This is the base class for block versions of the sparsity pattern and
 * dynamic sparsity pattern classes. It has not much functionality, but only
 * administrates an array of sparsity pattern objects and delegates work to
 * them. It has mostly the same interface as has the SparsityPattern, and
 * DynamicSparsityPattern, and simply transforms calls to its member functions
 * to calls to the respective member functions of the member sparsity
 * patterns.
 *
 * The largest difference between the SparsityPattern and
 * DynamicSparsityPattern classes and this class is that mostly, the matrices
 * have different properties and you will want to work on the blocks making up
 * the matrix rather than the whole matrix. You can access the different
 * blocks using the <tt>block(row,col)</tt> function.
 *
 * Attention: this object is not automatically notified if the size of one of
 * its subobjects' size is changed. After you initialize the sizes of the
 * subobjects, you will therefore have to call the <tt>collect_sizes()</tt>
 * function of this class! Note that, of course, all sub-matrices in a
 * (block-)row have to have the same number of rows, and that all sub-matrices
 * in a (block-)column have to have the same number of columns.
 *
 * You will in general not want to use this class, but one of the derived
 * classes.
 *
 * @todo Handle optimization of diagonal elements of the underlying
 * SparsityPattern correctly.
 *
 * @see
 * @ref GlossBlockLA "Block (linear algebra)"
 */
template <typename SparsityPatternType>
class BlockSparsityPatternBase : public SparsityPatternBase
{
public:
  /**
   * Declare type for container size.
   */
  using size_type = types::global_dof_index;

  /**
   * Define a value which is used to indicate that a certain value in the @p
   * colnums array is unused, i.e. does not represent a certain column number
   * index.
   *
   * This value is only an alias to the respective value of the
   * SparsityPattern class.
   */
  static const size_type invalid_entry = SparsityPattern::invalid_entry;

  /**
   * Initialize the matrix empty, that is with no memory allocated. This is
   * useful if you want such objects as member variables in other classes. You
   * can make the structure usable by calling the reinit() function.
   */
  BlockSparsityPatternBase();

  /**
   * Initialize the matrix with the given number of block rows and columns.
   * The blocks themselves are still empty, and you have to call
   * collect_sizes() after you assign them sizes.
   */
  BlockSparsityPatternBase(const size_type n_block_rows,
                           const size_type n_block_columns);

  /**
   * Copy constructor. This constructor is only allowed to be called if the
   * sparsity pattern to be copied is empty, i.e. there are no block allocated
   * at present. This is for the same reason as for the SparsityPattern, see
   * there for the details.
   */
  BlockSparsityPatternBase(const BlockSparsityPatternBase &bsp);

  /**
   * Resize the matrix, by setting the number of block rows and columns. This
   * deletes all blocks and replaces them with uninitialized ones, i.e. ones
   * for which also the sizes are not yet set. You have to do that by calling
   * the reinit() functions of the blocks themselves. Do not forget to call
   * collect_sizes() after that on this object.
   *
   * The reason that you have to set sizes of the blocks yourself is that the
   * sizes may be varying, the maximum number of elements per row may be
   * varying, etc. It is simpler not to reproduce the interface of the
   * SparsityPattern class here but rather let the user call whatever function
   * they desire.
   */
  void
  reinit(const size_type n_block_rows, const size_type n_block_columns);

  /**
   * Copy operator. For this the same holds as for the copy constructor: it is
   * declared, defined and fine to be called, but the latter only for empty
   * objects.
   */
  BlockSparsityPatternBase &
  operator=(const BlockSparsityPatternBase &);

  /**
   * This function collects the sizes of the sub-objects and stores them in
   * internal arrays, in order to be able to relay global indices into the
   * matrix to indices into the subobjects. You *must* call this function each
   * time after you have changed the size of the sub-objects.
   */
  void
  collect_sizes();

  /**
   * Access the block with the given coordinates.
   */
  SparsityPatternType &
  block(const size_type row, const size_type column);

  /**
   * Access the block with the given coordinates. Version for constant
   * objects.
   */
  const SparsityPatternType &
  block(const size_type row, const size_type column) const;

  /**
   * Grant access to the object describing the distribution of row indices to
   * the individual blocks.
   */
  const BlockIndices &
  get_row_indices() const;

  /**
   * Grant access to the object describing the distribution of column indices
   * to the individual blocks.
   */
  const BlockIndices &
  get_column_indices() const;

  /**
   * This function compresses the sparsity structures that this object
   * represents. It simply calls @p compress for all sub-objects.
   */
  void
  compress();

  /**
   * Return the number of blocks in a column.
   */
  size_type
  n_block_rows() const;

  /**
   * Return the number of blocks in a row.
   */
  size_type
  n_block_cols() const;

  /**
   * Return whether the object is empty. It is empty if no memory is
   * allocated, which is the same as that both dimensions are zero. This
   * function is just the concatenation of the respective call to all
   * sub-matrices.
   */
  bool
  empty() const;

  /**
   * Return the maximum number of entries per row. It returns the maximal
   * number of entries per row accumulated over all blocks in a row, and the
   * maximum over all rows.
   */
  size_type
  max_entries_per_row() const;

  /**
   * Add a nonzero entry to the matrix. This function may only be called for
   * non-compressed sparsity patterns.
   *
   * If the entry already exists, nothing bad happens.
   *
   * This function simply finds out to which block <tt>(i,j)</tt> belongs and
   * then relays to that block.
   */
  void
  add(const size_type i, const size_type j);

  /**
   * Add several nonzero entries to the specified matrix row.  This function
   * may only be called for non-compressed sparsity patterns.
   *
   * If some of the entries already exist, nothing bad happens.
   *
   * This function simply finds out to which blocks <tt>(row,col)</tt> for
   * <tt>col</tt> in the iterator range belong and then relays to those
   * blocks.
   */
  template <typename ForwardIterator>
  void
  add_entries(const size_type row,
              ForwardIterator begin,
              ForwardIterator end,
              const bool      indices_are_sorted = false);

  /**
   * Add several nonzero entries to the specified matrix row. This function may
   * only be called for non-compressed sparsity patterns and works the same way
   * as the overload which takes iterators.
   */
  virtual void
  add_row_entries(const size_type                  &row,
                  const ArrayView<const size_type> &columns,
                  const bool indices_are_sorted = false) override;

  using SparsityPatternBase::add_entries;

  /**
   * Return number of rows of this matrix, which equals the dimension of the
   * image space. It is the sum of rows of the (block-)rows of sub-matrices.
   */
  using SparsityPatternBase::n_rows;

  /**
   * Return number of columns of this matrix, which equals the dimension of
   * the range space. It is the sum of columns of the (block-)columns of
   * sub-matrices.
   */
  using SparsityPatternBase::n_cols;

  /**
   * Check if a value at a certain position may be non-zero.
   */
  bool
  exists(const size_type i, const size_type j) const;

  /**
   * Number of entries in a specific row, added up over all the blocks that
   * form this row.
   */
  unsigned int
  row_length(const size_type row) const;

  /**
   * Return the number of nonzero elements of this matrix. Actually, it
   * returns the number of entries in the sparsity pattern; if any of the
   * entries should happen to be zero, it is counted anyway.
   *
   * This function may only be called if the matrix struct is compressed. It
   * does not make too much sense otherwise anyway.
   *
   * In the present context, it is the sum of the values as returned by the
   * sub-objects.
   */
  size_type
  n_nonzero_elements() const;

  /**
   * Print the sparsity of the matrix. The output consists of one line per row
   * of the format <tt>[i,j1,j2,j3,...]</tt>. <i>i</i> is the row number and
   * <i>jn</i> are the allocated columns in this row.
   */
  void
  print(std::ostream &out) const;

  /**
   * Print the sparsity of the matrix in a format that <tt>gnuplot</tt>
   * understands and which can be used to plot the sparsity pattern in a
   * graphical way. This is the same functionality implemented for usual
   * sparsity patterns, see
   * SparsityPattern::print_gnuplot().
   */
  void
  print_gnuplot(std::ostream &out) const;

  /**
   * Print the sparsity of the matrix in <tt>svg</tt> format. This is the same
   * functionality implemented for usual sparsity patterns, see
   * SparsityPattern::print_svg().
   */
  void
  print_svg(std::ostream &out) const;

  /**
   * Determine an estimate for the memory consumption (in bytes) of this
   * object.
   */
  std::size_t
  memory_consumption() const;

  /**
   * @addtogroup Exceptions
   * @{
   */

  /**
   * Exception
   */
  DeclExceptionMsg(
    ExcNeedsCollectSizes,
    "The number of rows and columns (returned by n_rows() and n_cols()) does "
    "not match their directly computed values. This typically means that a "
    "call to collect_sizes() is missing.");

  /**
   * Exception
   */
  DeclException4(ExcIncompatibleRowNumbers,
                 int,
                 int,
                 int,
                 int,
                 << "The blocks [" << arg1 << ',' << arg2 << "] and [" << arg3
                 << ',' << arg4 << "] have differing row numbers.");
  /**
   * Exception
   */
  DeclException4(ExcIncompatibleColNumbers,
                 int,
                 int,
                 int,
                 int,
                 << "The blocks [" << arg1 << ',' << arg2 << "] and [" << arg3
                 << ',' << arg4 << "] have differing column numbers.");
  /** @} */

protected:
  /**
   * Number of block rows.
   */
  size_type block_rows;

  /**
   * Number of block columns.
   */
  size_type block_columns;

  /**
   * Array of sparsity patterns.
   */
  Table<2, std::unique_ptr<SparsityPatternType>> sub_objects;

  /**
   * Object storing and managing the transformation of row indices to indices
   * of the sub-objects.
   */
  BlockIndices row_indices;

  /**
   * Object storing and managing the transformation of column indices to
   * indices of the sub-objects.
   */
  BlockIndices column_indices;

private:
  /**
   * Internal utility function for computing the number of rows.
   */
  size_type
  compute_n_rows() const;

  /**
   * Internal utility function for computing the number of columns.
   */
  size_type
  compute_n_cols() const;

  /**
   * Temporary vector for counting the elements written into the individual
   * blocks when doing a collective add or set.
   */
  std::vector<size_type> counter_within_block;

  /**
   * Temporary vector for column indices on each block when writing local to
   * global data on each sparse matrix.
   */
  std::vector<std::vector<size_type>> block_column_indices;

  // Make the block sparse matrix a friend, so that it can use our
  // #row_indices and #column_indices objects.
  template <typename number>
  friend class BlockSparseMatrix;
};



/**
 * This class extends the base class to implement an array of sparsity
 * patterns that can be used by block sparse matrix objects. It only adds a
 * few additional member functions, but the main interface stems from the base
 * class, see there for more information.
 *
 * This class is an example of the "static" type of
 * @ref Sparsity.
 */
class BlockSparsityPattern : public BlockSparsityPatternBase<SparsityPattern>
{
public:
  /**
   * Initialize the matrix empty, that is with no memory allocated. This is
   * useful if you want such objects as member variables in other classes. You
   * can make the structure usable by calling the reinit() function.
   */
  BlockSparsityPattern() = default;

  /**
   * Initialize the matrix with the given number of block rows and columns.
   * The blocks themselves are still empty, and you have to call
   * collect_sizes() after you assign them sizes.
   */
  BlockSparsityPattern(const size_type n_rows, const size_type n_columns);

  /**
   * Forwarding to BlockSparsityPatternBase::reinit().
   */
  void
  reinit(const size_type n_block_rows, const size_type n_block_columns);

  /**
   * Initialize the pattern with two BlockIndices for the block structures of
   * matrix rows and columns as well as a row length vector.
   *
   * The row length vector should be in the format produced by DoFTools.
   * Alternatively, there is a simplified version, where each of the inner
   * vectors has length one. Then, the corresponding entry is used as the
   * maximal row length.
   *
   * For the diagonal blocks, the inner SparsityPattern is initialized with
   * optimized diagonals, while this is not done for the off-diagonal blocks.
   */
  void
  reinit(const BlockIndices                           &row_indices,
         const BlockIndices                           &col_indices,
         const std::vector<std::vector<unsigned int>> &row_lengths);


  /**
   * Return whether the structure is compressed or not, i.e. whether all
   * sub-matrices are compressed.
   */
  bool
  is_compressed() const;

  /**
   * Copy data from an object of type BlockDynamicSparsityPattern, i.e. resize
   * this object to the size of the given argument, and copy over the contents
   * of each of the subobjects. Previous content of this object is lost.
   */
  void
  copy_from(const BlockDynamicSparsityPattern &dsp);
};



/**
 * This class extends the base class to implement an array of compressed
 * sparsity patterns that can be used to initialize objects of type
 * BlockSparsityPattern. It does not add additional member functions, but
 * rather acts as an @p alias to introduce the name of this class, without
 * requiring the user to specify the templated name of the base class. For
 * information on the interface of this class refer to the base class. The
 * individual blocks are based on the DynamicSparsityPattern class.
 *
 * This class is an example of the "dynamic" type of
 * @ref Sparsity.
 *
 * <h3>Example</h3>
 *
 * Usage of this class is very similar to DynamicSparsityPattern, but since
 * the use of block indices causes some additional complications, we give a
 * short example.
 *
 * After the DoFHandler <tt>dof</tt> and the AffineConstraints
 * <tt>constraints</tt> have been set up with a system element, we must count
 * the degrees of freedom in each matrix block:
 *
 * @code
 * const std::vector<unsigned int> dofs_per_block =
 *   DoFTools::count_dofs_per_fe_block(dof);
 * @endcode
 *
 * Now, we are ready to set up the BlockDynamicSparsityPattern.
 *
 * @code
 * BlockDynamicSparsityPattern dsp(fe.n_blocks(), fe.n_blocks());
 * for (unsigned int i = 0; i < fe.n_blocks(); ++i)
 *   for (unsigned int j = 0; j < fe.n_blocks(); ++j)
 *     dsp.block(i, j).reinit(dofs_per_block[i], dofs_per_block[j]);
 * dsp.collect_sizes();
 * @endcode
 *
 * It is filled as if it were a normal pattern
 *
 * @code
 * DoFTools::make_sparsity_pattern(dof, dsp);
 * constraints.condense(dsp);
 * @endcode
 *
 * In the end, it is copied to a normal BlockSparsityPattern for later use.
 *
 * @code
 * BlockSparsityPattern sparsity;
 * sparsity.copy_from(dsp);
 * @endcode
 */

class BlockDynamicSparsityPattern
  : public BlockSparsityPatternBase<DynamicSparsityPattern>
{
public:
  /**
   * Initialize the matrix empty, that is with no memory allocated. This is
   * useful if you want such objects as member variables in other classes. You
   * can make the structure usable by calling the reinit() function.
   */
  BlockDynamicSparsityPattern() = default;

  /**
   * Initialize the matrix with the given number of block rows and columns.
   * The blocks themselves are still empty, and you have to call
   * collect_sizes() after you assign them sizes.
   */
  BlockDynamicSparsityPattern(const size_type n_rows,
                              const size_type n_columns);

  /**
   * Initialize the pattern with two BlockIndices for the block structures of
   * matrix rows and columns. This function is equivalent to calling the
   * previous constructor with the length of the two index vector and then
   * entering the index values.
   */
  BlockDynamicSparsityPattern(const std::vector<size_type> &row_block_sizes,
                              const std::vector<size_type> &col_block_sizes);

  /**
   * Initialize the pattern with symmetric blocks. The number of IndexSets in
   * the vector determine the number of rows and columns of blocks. The size
   * of each block is determined by the size() of the respective IndexSet.
   * Each block only stores the rows given by the values in the IndexSet,
   * which is useful for distributed memory parallel computations and usually
   * corresponds to the locally relevant DoFs.
   */
  BlockDynamicSparsityPattern(const std::vector<IndexSet> &partitioning);

  /**
   * Initialize the pattern with two BlockIndices for the block structures of
   * matrix rows and columns.
   */
  BlockDynamicSparsityPattern(const BlockIndices &row_indices,
                              const BlockIndices &col_indices);


  /**
   * Resize the pattern to a tensor product of matrices with dimensions
   * defined by the arguments.
   *
   * The matrix will have as many block rows and columns as there are entries
   * in the two arguments. The block at position (<i>i,j</i>) will have the
   * dimensions <tt>row_block_sizes[i]</tt> times <tt>col_block_sizes[j]</tt>.
   */
  void
  reinit(const std::vector<size_type> &row_block_sizes,
         const std::vector<size_type> &col_block_sizes);

  /**
   * Resize the pattern with symmetric blocks determined by the size() of each
   * IndexSet. See the constructor taking a vector of IndexSets for details.
   */
  void
  reinit(const std::vector<IndexSet> &partitioning);

  /**
   * Resize the matrix to a tensor product of matrices with dimensions defined
   * by the arguments. The two BlockIndices objects must be initialized and
   * the sparsity pattern will have the same block structure afterwards.
   */
  void
  reinit(const BlockIndices &row_indices, const BlockIndices &col_indices);

  /**
   * Access to column number field. Return the column number of the @p index
   * th entry in row @p row.
   */
  size_type
  column_number(const size_type row, const unsigned int index) const;

  /**
   * Allow the use of the reinit functions of the base class as well.
   */
  using BlockSparsityPatternBase<DynamicSparsityPattern>::reinit;
};

/** @} */


#ifdef DEAL_II_WITH_TRILINOS


namespace TrilinosWrappers
{
  /**
   * @addtogroup TrilinosWrappers
   * @{
   */

  /**
   * This class extends the base class to implement an array of Trilinos
   * sparsity patterns that can be used to initialize Trilinos block sparse
   * matrices that can be distributed among different processors. It is used in
   * the same way as the dealii::BlockSparsityPattern except that it builds upon
   * the TrilinosWrappers::SparsityPattern instead of the
   * dealii::SparsityPattern.
   *
   * This class is has properties of the "dynamic" type of
   * @ref Sparsity
   * (in the sense that it can extend the memory if too little elements were
   * allocated), but otherwise is more like the basic deal.II SparsityPattern
   * (in the sense that the method compress() needs to be called before the
   * pattern can be used).
   *
   * This class is used in step-32.
   */
  class BlockSparsityPattern
    : public dealii::BlockSparsityPatternBase<SparsityPattern>
  {
  public:
    /**
     * Initialize the matrix empty, that is with no memory allocated. This is
     * useful if you want such objects as member variables in other classes.
     * You can make the structure usable by calling the reinit() function.
     */
    BlockSparsityPattern() = default;

    /**
     * Initialize the matrix with the given number of block rows and columns.
     * The blocks themselves are still empty, and you have to call
     * collect_sizes() after you assign them sizes.
     */
    BlockSparsityPattern(const size_type n_rows, const size_type n_columns);

    /**
     * Initialize the pattern with two BlockIndices for the block structures
     * of matrix rows and columns. This function is equivalent to calling the
     * previous constructor with the length of the two index vector and then
     * entering the index values.
     */
    BlockSparsityPattern(const std::vector<size_type> &row_block_sizes,
                         const std::vector<size_type> &col_block_sizes);

    /**
     * Initialize the pattern with an array of index sets that specifies both
     * rows and columns of the matrix (so the final matrix will be a square
     * matrix), where the size() of the IndexSets specifies the size of the
     * blocks and the values in each IndexSet denotes the rows that are going
     * to be saved in each block.
     */
    BlockSparsityPattern(const std::vector<IndexSet> &parallel_partitioning,
                         const MPI_Comm communicator = MPI_COMM_WORLD);

    /**
     * Initialize the pattern with two arrays of index sets that specify rows
     * and columns of the matrix, where the size() of the IndexSets specifies
     * the size of the blocks and the values in each IndexSet denotes the rows
     * that are going to be saved in each block. The additional index set
     * writable_rows is used to set all rows that we allow to write locally.
     * This constructor is used to create matrices that allow several threads
     * to write simultaneously into the matrix (to different rows, of course),
     * see the method TrilinosWrappers::SparsityPattern::reinit method with
     * three index set arguments for more details.
     */
    BlockSparsityPattern(
      const std::vector<IndexSet> &row_parallel_partitioning,
      const std::vector<IndexSet> &column_parallel_partitioning,
      const std::vector<IndexSet> &writeable_rows,
      const MPI_Comm               communicator = MPI_COMM_WORLD);

    /**
     * Resize the matrix to a tensor product of matrices with dimensions
     * defined by the arguments.
     *
     * The matrix will have as many block rows and columns as there are
     * entries in the two arguments. The block at position (<i>i,j</i>) will
     * have the dimensions <tt>row_block_sizes[i]</tt> times
     * <tt>col_block_sizes[j]</tt>.
     */
    void
    reinit(const std::vector<size_type> &row_block_sizes,
           const std::vector<size_type> &col_block_sizes);

    /**
     * Resize the matrix to a square tensor product of matrices. See the
     * constructor that takes a vector of IndexSets for details.
     */
    void
    reinit(const std::vector<IndexSet> &parallel_partitioning,
           const MPI_Comm               communicator = MPI_COMM_WORLD);

    /**
     * Resize the matrix to a rectangular block matrices. This method allows
     * rows and columns to be different, both in the outer block structure and
     * within the blocks.
     */
    void
    reinit(const std::vector<IndexSet> &row_parallel_partitioning,
           const std::vector<IndexSet> &column_parallel_partitioning,
           const MPI_Comm               communicator = MPI_COMM_WORLD);

    /**
     * Resize the matrix to a rectangular block matrices that furthermore
     * explicitly specify the writable rows in each of the blocks. This method
     * is used to create matrices that allow several threads to write
     * simultaneously into the matrix (to different rows, of course), see the
     * method TrilinosWrappers::SparsityPattern::reinit method with three
     * index set arguments for more details.
     */
    void
    reinit(const std::vector<IndexSet> &row_parallel_partitioning,
           const std::vector<IndexSet> &column_parallel_partitioning,
           const std::vector<IndexSet> &writeable_rows,
           const MPI_Comm               communicator = MPI_COMM_WORLD);

    /**
     * Allow the use of the reinit functions of the base class as well.
     */
    using BlockSparsityPatternBase<SparsityPattern>::reinit;
  };

  /** @} */

} /* namespace TrilinosWrappers */

#endif

/*--------------------- Template functions ----------------------------------*/



template <typename SparsityPatternType>
inline SparsityPatternType &
BlockSparsityPatternBase<SparsityPatternType>::block(const size_type row,
                                                     const size_type column)
{
  AssertIndexRange(row, n_block_rows());
  AssertIndexRange(column, n_block_cols());
  return *sub_objects(row, column);
}



template <typename SparsityPatternType>
inline const SparsityPatternType &
BlockSparsityPatternBase<SparsityPatternType>::block(
  const size_type row,
  const size_type column) const
{
  AssertIndexRange(row, n_block_rows());
  AssertIndexRange(column, n_block_cols());
  return *sub_objects(row, column);
}



template <typename SparsityPatternType>
inline const BlockIndices &
BlockSparsityPatternBase<SparsityPatternType>::get_row_indices() const
{
  return row_indices;
}



template <typename SparsityPatternType>
inline const BlockIndices &
BlockSparsityPatternBase<SparsityPatternType>::get_column_indices() const
{
  return column_indices;
}



template <typename SparsityPatternType>
inline void
BlockSparsityPatternBase<SparsityPatternType>::add(const size_type i,
                                                   const size_type j)
{
  // if you get an error here, are
  // you sure you called
  // <tt>collect_sizes()</tt> before?
  const std::pair<size_type, size_type> row_index =
                                          row_indices.global_to_local(i),
                                        col_index =
                                          column_indices.global_to_local(j);
  sub_objects[row_index.first][col_index.first]->add(row_index.second,
                                                     col_index.second);
}



template <typename SparsityPatternType>
template <typename ForwardIterator>
void
BlockSparsityPatternBase<SparsityPatternType>::add_entries(
  const size_type row,
  ForwardIterator begin,
  ForwardIterator end,
  const bool      indices_are_sorted)
{
  // In debug mode, verify that collect_sizes() was called by redoing the
  // calculation
  Assert(n_rows() == compute_n_rows(), ExcNeedsCollectSizes());
  Assert(n_cols() == compute_n_cols(), ExcNeedsCollectSizes());

  const size_type n_cols = static_cast<size_type>(end - begin);

  if (indices_are_sorted && n_cols > 0)
    {
      block_column_indices[0].resize(0);

      const std::pair<size_type, size_type> row_index =
        this->row_indices.global_to_local(row);
      const auto n_blocks = column_indices.size();

      // Assume we start with the first block: since we assemble sparsity
      // patterns one cell at a time this should always be true
      size_type current_block       = 0;
      size_type current_start_index = column_indices.block_start(current_block);
      size_type next_start_index =
        current_block == n_blocks - 1 ?
          numbers::invalid_dof_index :
          column_indices.block_start(current_block + 1);

      for (auto it = begin; it < end; ++it)
        {
          // BlockIndices::global_to_local() is a bit slow so instead we just
          // keep track of which block we are in - as the indices are sorted we
          // know that the block number can only increase.
          if (*it >= next_start_index)
            {
              // we found a column outside the present block: write the present
              // block entries and continue to the next block
              sub_objects[row_index.first][current_block]->add_entries(
                row_index.second,
                block_column_indices[0].begin(),
                block_column_indices[0].end(),
                true);
              block_column_indices[0].clear();

              auto block_and_col  = column_indices.global_to_local(*it);
              current_block       = block_and_col.first;
              current_start_index = column_indices.block_start(current_block);
              next_start_index =
                current_block == n_blocks - 1 ?
                  numbers::invalid_dof_index :
                  column_indices.block_start(current_block + 1);
            }
          const size_type local_index = *it - current_start_index;
          block_column_indices[0].push_back(local_index);

          // Check that calculation:
          if constexpr (running_in_debug_mode())
            {
              {
                auto check_block_and_col = column_indices.global_to_local(*it);
                Assert(current_block == check_block_and_col.first,
                       ExcInternalError());
                Assert(local_index == check_block_and_col.second,
                       ExcInternalError());
              }
            }
        }
      // add whatever is left over:
      sub_objects[row_index.first][current_block]->add_entries(
        row_index.second,
        block_column_indices[0].begin(),
        block_column_indices[0].end(),
        true);

      return;
    }
  else
    {
      // Resize sub-arrays to n_cols. This
      // is a bit wasteful, but we resize
      // only a few times (then the maximum
      // row length won't increase that
      // much any more). At least we know
      // that all arrays are going to be of
      // the same size, so we can check
      // whether the size of one is large
      // enough before actually going
      // through all of them.
      if (block_column_indices[0].size() < n_cols)
        for (size_type i = 0; i < this->n_block_cols(); ++i)
          block_column_indices[i].resize(n_cols);

      // Reset the number of added elements
      // in each block to zero.
      for (size_type i = 0; i < this->n_block_cols(); ++i)
        counter_within_block[i] = 0;

      // Go through the column indices to
      // find out which portions of the
      // values should be set in which
      // block of the matrix. We need to
      // touch all the data, since we can't
      // be sure that the data of one block
      // is stored contiguously (in fact,
      // indices will be intermixed when it
      // comes from an element matrix).
      for (ForwardIterator it = begin; it != end; ++it)
        {
          const size_type col = *it;

          const std::pair<size_type, size_type> col_index =
            this->column_indices.global_to_local(col);

          const size_type local_index = counter_within_block[col_index.first]++;

          block_column_indices[col_index.first][local_index] = col_index.second;
        }

      // Now we found out about where the
      // individual columns should start and
      // where we should start reading out
      // data. Now let's write the data into
      // the individual blocks!
      const std::pair<size_type, size_type> row_index =
        this->row_indices.global_to_local(row);
      for (size_type block_col = 0; block_col < n_block_cols(); ++block_col)
        {
          if (counter_within_block[block_col] == 0)
            continue;
          sub_objects[row_index.first][block_col]->add_entries(
            row_index.second,
            block_column_indices[block_col].begin(),
            block_column_indices[block_col].begin() +
              counter_within_block[block_col],
            indices_are_sorted);
        }
    }
}



template <typename SparsityPatternType>
void
BlockSparsityPatternBase<SparsityPatternType>::add_row_entries(
  const size_type                  &row,
  const ArrayView<const size_type> &columns,
  const bool                        indices_are_sorted)
{
  add_entries(row, columns.begin(), columns.end(), indices_are_sorted);
}



template <typename SparsityPatternType>
inline bool
BlockSparsityPatternBase<SparsityPatternType>::exists(const size_type i,
                                                      const size_type j) const
{
  // if you get an error here, are
  // you sure you called
  // <tt>collect_sizes()</tt> before?
  const std::pair<size_type, size_type> row_index =
                                          row_indices.global_to_local(i),
                                        col_index =
                                          column_indices.global_to_local(j);
  return sub_objects[row_index.first][col_index.first]->exists(
    row_index.second, col_index.second);
}



template <typename SparsityPatternType>
inline unsigned int
BlockSparsityPatternBase<SparsityPatternType>::row_length(
  const size_type row) const
{
  const std::pair<size_type, size_type> row_index =
    row_indices.global_to_local(row);

  unsigned int c = 0;

  for (size_type b = 0; b < n_block_rows(); ++b)
    c += sub_objects[row_index.first][b]->row_length(row_index.second);

  return c;
}



template <typename SparsityPatternType>
inline typename BlockSparsityPatternBase<SparsityPatternType>::size_type
BlockSparsityPatternBase<SparsityPatternType>::n_block_cols() const
{
  return block_columns;
}



template <typename SparsityPatternType>
inline typename BlockSparsityPatternBase<SparsityPatternType>::size_type
BlockSparsityPatternBase<SparsityPatternType>::n_block_rows() const
{
  return block_rows;
}


inline BlockDynamicSparsityPattern::size_type
BlockDynamicSparsityPattern::column_number(const size_type    row,
                                           const unsigned int index) const
{
  // .first= ith block, .second = jth row in that block
  const std::pair<size_type, size_type> row_index =
    row_indices.global_to_local(row);

  AssertIndexRange(index, row_length(row));

  size_type c             = 0;
  size_type block_columns = 0; // sum of n_cols for all blocks to the left
  for (unsigned int b = 0; b < this->n_block_cols(); ++b)
    {
      unsigned int rowlen =
        sub_objects[row_index.first][b]->row_length(row_index.second);
      if (index < c + rowlen)
        return block_columns +
               sub_objects[row_index.first][b]->column_number(row_index.second,
                                                              index - c);
      c += rowlen;
      block_columns += sub_objects[row_index.first][b]->n_cols();
    }

  DEAL_II_ASSERT_UNREACHABLE();
  return 0;
}


inline void
BlockSparsityPattern::reinit(const size_type new_block_rows,
                             const size_type new_block_columns)
{
  BlockSparsityPatternBase<SparsityPattern>::reinit(new_block_rows,
                                                    new_block_columns);
}


DEAL_II_NAMESPACE_CLOSE

#endif
