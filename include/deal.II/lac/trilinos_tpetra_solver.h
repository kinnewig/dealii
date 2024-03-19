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

#ifndef dealii_trilinos_tpetra_solver_h
#define dealii_trilinos_tpetra_solver_h


#include <deal.II/base/config.h>

#ifdef DEAL_II_TRILINOS_WITH_TPETRA


#  include <deal.II/lac/solver_control.h>
#  include <deal.II/lac/trilinos_tpetra_precondition.h>
#  include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>
#  include <deal.II/lac/trilinos_tpetra_vector.h>

//  Belos
#  include <BelosSolverFactory.hpp>
#  include <BelosTpetraAdapter.hpp>

// Ifpack2
#  include <Ifpack2_Factory.hpp>

// Teuchos
#  include <Teuchos_RCP.hpp>

#  include <memory>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace TpetraWrappers
  {
// forward declarations
#  ifndef DOXYGEN
    template <typename Number, typename Node>
    class SparseMatrix;
#  endif



    template <
      typename Number,
      typename Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType,
      typename LinearOperator = Tpetra::
        Operator<Number, int, dealii::types::signed_global_dof_index, Node>,
      typename MultiVector = Tpetra::
        MultiVector<Number, int, dealii::types::signed_global_dof_index, Node>>
    class SolverBase
    {
    public:
      using value_type = Number;

      /**
       * Enumeration object that is set in the constructor of the derived
       * classes and tells Trilinos which solver to use. This option can also be
       * set in the user program, so one might use this base class instead of
       * one of the specialized derived classes when the solver should be set at
       * runtime. Currently enabled options are:
       */
      enum SolverName
      {
        /**
         * Use the conjugate gradient (CG) algorithm.
         */
        cg,
        /**
         * Use the conjugate gradient squared (CGS) algorithm.
         */
        cgs,
        /**
         * Use the generalized minimum residual (GMRES) algorithm.
         */
        gmres,
        /**
         * Use the biconjugate gradient stabilized (BICGStab) algorithm.
         */
        bicgstab,
        /**
         * Use the transpose-free quasi-minimal residual (TFQMR) method.
         */
        tfqmr
      } solver_name;

      /**
       * Constructor. Takes a Teuchos::RCP to a Teuchos::ParameterList
       */
      SolverBase(SolverControl                             &cn,
                 const Teuchos::RCP<Teuchos::ParameterList> pm);

      /**
       * Destructor.
       */
      virtual ~SolverBase() = default;

      /**
       * Access to object that controls convergence.
       */
      SolverControl &
      control() const;

      /**
       * Number of Iterations
       */
      unsigned int num_iterations;

      /**
       * Exception
       */
      DeclException1(ExcTrilinosError,
                     int,
                     << "An error with error number " << arg1
                     << " occurred while calling a Trilinos function");

    protected:
      /**
       * Reference to the object that controls convergence of the iterative
       * solver. In fact, for these Trilinos wrappers, Trilinos does so itself,
       * but we copy the data from this object before starting the solution
       * process, and copy the data back into it afterwards.
       */
      SolverControl &solver_control;

      template <typename Preconditioner>
      void
      do_solve(const Preconditioner &preconditioner);

      /**
       * Teuchos::RCP pointer to Teuchos::ParameterList
       */
      Teuchos::RCP<Teuchos::ParameterList> parameter_list;

      /**
       * A structure that collects the Trilinos sparse matrix, the right hand
       * side vector and the solution vector, which is passed down to the
       * Trilinos solver.
       */
      Teuchos::RCP<Belos::LinearProblem<Number, MultiVector, LinearOperator>>
        linear_problem;

      /**
       * A structure that contains the Trilinos solver and preconditioner
       * objects.
       */
      Teuchos::RCP<Belos::SolverManager<Number, MultiVector, LinearOperator>>
        solver;

    }; // class SolverBase



    template <typename Number = double,
              typename Node =
                Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class SolverTpetra
      : public SolverBase<
          Number,
          Node,
          Tpetra::
            Operator<Number, int, dealii::types::signed_global_dof_index, Node>,
          Tpetra::MultiVector<Number,
                              int,
                              dealii::types::signed_global_dof_index,
                              Node>>
    {
    public:
      using VectorType = Tpetra::
        Vector<Number, int, dealii::types::signed_global_dof_index, Node>;

      using MultiVectorType = Tpetra::
        MultiVector<Number, int, dealii::types::signed_global_dof_index, Node>;

      using LinearOperatorType = Tpetra::
        Operator<Number, int, dealii::types::signed_global_dof_index, Node>;

      SolverTpetra(SolverControl                             &cn,
                   const Teuchos::RCP<Teuchos::ParameterList> pm);

      void
      solve(const SparseMatrix<Number>              &A,
            Vector<Number>                          &x,
            Vector<Number>                          &b,
            const PreconditionBase<Number,
                                   Node,
                                   LinearOperatorType,
                                   MultiVectorType> &preconditioner);

    private:
      using SolverBase<Number, Node, LinearOperatorType, MultiVectorType>::
        parameter_list;
      using SolverBase<Number, Node, LinearOperatorType, MultiVectorType>::
        linear_problem;
      using SolverBase<Number, Node, LinearOperatorType, MultiVectorType>::
        solver;
      using SolverBase<Number, Node, LinearOperatorType, MultiVectorType>::
        do_solve;

    }; // class SolverTpetra



    template <typename Number = double,
              typename Node =
                Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class SolverXpetra
      : public SolverBase<
          Number,
          Node,
          Belos::OperatorT<
            Xpetra::MultiVector<Number,
                                int,
                                dealii::types::signed_global_dof_index,
                                Node>>,
          Xpetra::MultiVector<Number,
                              int,
                              dealii::types::signed_global_dof_index,
                              Node>>
    {
    public:
      using VectorType = Tpetra::
        Vector<Number, int, dealii::types::signed_global_dof_index, Node>;

      using MultiVectorType = Xpetra::
        MultiVector<Number, int, dealii::types::signed_global_dof_index, Node>;

      using LinearOperatorType = Belos::OperatorT<MultiVectorType>;

      using XpetraOpType = Belos::
        XpetraOp<Number, int, dealii::types::signed_global_dof_index, Node>;

      using XpetraTpetraMultiVectorType =
        Xpetra::TpetraMultiVector<Number,
                                  int,
                                  dealii::types::signed_global_dof_index,
                                  Node>;

      using XpetraTpetraCrsMatrixType =
        Xpetra::TpetraCrsMatrix<Number,
                                int,
                                dealii::types::signed_global_dof_index,
                                Node>;

      using XpetraCrsMatrixType = Xpetra::
        CrsMatrix<Number, int, dealii::types::signed_global_dof_index, Node>;

      using XpetraCrsMatrixWrapType =
        Xpetra::CrsMatrixWrap<Number,
                              int,
                              dealii::types::signed_global_dof_index,
                              Node>;

      using XpetraMatrixType = Xpetra::
        Matrix<Number, int, dealii::types::signed_global_dof_index, Node>;

      /**
       * Constructor. Takes a Teuchos::RCP to a Teuchos::ParameterList
       */
      SolverXpetra(SolverControl                             &cn,
                   const Teuchos::RCP<Teuchos::ParameterList> pm);

      /**
       * Solve the linear system <tt>Ax=b</tt>. Depending on the information
       * provided by derived classes and the object passed as a preconditioner,
       * one of the linear solvers and preconditioners of Trilinos is chosen.
       */
      void
      solve(const SparseMatrix<Number>              &A,
            Vector<Number>                          &x,
            Vector<Number>                          &b,
            const PreconditionBase<Number,
                                   Node,
                                   LinearOperatorType,
                                   MultiVectorType> &preconditioner);

    private:
      using SolverBase<Number, Node, LinearOperatorType, MultiVectorType>::
        parameter_list;
      using SolverBase<Number, Node, LinearOperatorType, MultiVectorType>::
        linear_problem;
      using SolverBase<Number, Node, LinearOperatorType, MultiVectorType>::
        solver;
      using SolverBase<Number, Node, LinearOperatorType, MultiVectorType>::
        do_solve;

    }; // class SolverXpetra

  } // namespace TpetraWrappers
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_TPETRA

#endif // DEALII_TRILINOS_TPETRA_SOLVER_H
