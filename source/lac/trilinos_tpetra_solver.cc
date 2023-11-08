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

#include <deal.II/base/config.h>

#ifdef DEAL_II_TRILINOS_WITH_TPETRA

#  include <deal.II/base/conditional_ostream.h>

#  include <deal.II/lac/affine_constraints.h>
#  include <deal.II/lac/trilinos_tpetra_solver.h>
#  include <deal.II/lac/trilinos_tpetra_vector.h>

#  include <Tpetra_Operator.hpp>

#  include <cmath>
#  include <limits>
#  include <memory>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace TpetraWrappers
  {

    template <typename Number,
              typename Node,
              typename LinearOperator,
              typename MultiVector>
    SolverBase<Number, Node, LinearOperator, MultiVector>::SolverBase(
      SolverControl                       &cn,
      Teuchos::RCP<Teuchos::ParameterList> pl)
      : parameter_list(pl)
      , solver_control(cn)
    {}



    template <typename Number,
              typename Node,
              typename LinearOperator,
              typename MultiVector>
    SolverControl &
    SolverBase<Number, Node, LinearOperator, MultiVector>::control() const
    {
      return solver_control;
    }



    template <typename Number,
              typename Node,
              typename LinearOperator,
              typename MultiVector>
    template <typename Preconditioner>
    void
    SolverBase<Number, Node, LinearOperator, MultiVector>::do_solve(
      const Preconditioner &preconditioner)
    {
      linear_problem->setRightPrec(preconditioner.trilinos_rcp());
      linear_problem->setProblem();

      solver->setProblem(linear_problem);

      // Attempt to solve the linear system.  result == Belos::Converged
      // means that it was solved to the desired tolerance.  This call
      // overwrites X with the computed approximate solution.
      Belos::ReturnType result = solver->solve();

      // Ask the solver how many iterations the last solve() took.
      const int numIters = solver->getNumIters();
    }



    template <typename Number, typename Node>
    SolverTpetra<Number, Node>::SolverTpetra(
      dealii::SolverControl                     &cn,
      const Teuchos::RCP<Teuchos::ParameterList> pm)
      : SolverBase<Number, Node, LinearOperatorType, MultiVectorType>(cn, pm)
    {}



    template <typename Number, typename Node>
    void
    SolverTpetra<Number, Node>::solve(
      const SparseMatrix<Number> &A,
      Vector<Number>             &x,
      Vector<Number>             &b,
      const PreconditionBase<Number, Node, LinearOperatorType, MultiVectorType>
        &preconditioner)
    {
      // TODO: BEGIN Parameterlist:
      //       We set the parameter for the preconditioner here
      // Make an empty new parameter list.
      parameter_list = Teuchos::parameterList();

      parameter_list->set("Num Blocks", 40);
      parameter_list->set("Maximum Iterations", 400);
      parameter_list->set("Convergence Tolerance", 1.0e-8);
      // TODO: END ParameterList

      // TODO: make the solver user selectable
      // the list of solver parameters created above.
      Belos::SolverFactory<Number, MultiVectorType, LinearOperatorType> factory;
      solver = factory.create("GMRES", parameter_list);

      // Create a LinearProblem struct with the problem to solve.
      linear_problem = Teuchos::rcp(
        new Belos::LinearProblem<Number, MultiVectorType, LinearOperatorType>(
          A.trilinos_rcp(), x.trilinos_rcp(), b.trilinos_rcp()));

      do_solve(preconditioner);
    }



    template <typename Number, typename Node>
    SolverXpetra<Number, Node>::SolverXpetra(
      dealii::SolverControl                     &cn,
      const Teuchos::RCP<Teuchos::ParameterList> pm)
      : SolverBase<Number, Node, LinearOperatorType, MultiVectorType>(cn, pm)
    {}


    template <typename Number, typename Node>
    void
    SolverXpetra<Number, Node>::solve(
      const SparseMatrix<Number> &A,
      Vector<Number>             &x,
      Vector<Number>             &b,
      const PreconditionBase<Number, Node, LinearOperatorType, MultiVectorType>
        &preconditioner)
    {
      // TODO: BEGIN Parameterlist:
      //       We set the parameter for the preconditioner here
      // Make an empty new parameter list.
      parameter_list = Teuchos::parameterList();

      // parameter_list->set ("Num Blocks", 40);
      // parameter_list->set ("Maximum Iterations", 400);
      // parameter_list->set ("Convergence Tolerance", 1.0e-8);
      //  TODO: END ParameterList

      // TODO: make the solver user selectable
      Belos::SolverFactory<Number, MultiVectorType, LinearOperatorType> factory;
      solver = factory.create("CG", parameter_list);

      // Cast x and b to Xpetra
      Teuchos::RCP<MultiVectorType> xpetra_x =
        Teuchos::rcp(new XpetraTpetraMultiVectorType(x.trilinos_rcp()));
      Teuchos::RCP<MultiVectorType> xpetra_b =
        Teuchos::rcp(new XpetraTpetraMultiVectorType(b.trilinos_rcp()));

      // Cast A to Xpetra
      Teuchos::RCP<XpetraCrsMatrixType> xpetra_A =
        Teuchos::rcp(new XpetraTpetraCrsMatrixType(A.trilinos_rcp()));

      // Next we need to create a Belos::OperatorT obeject from xpetra_A
      Teuchos::RCP<XpetraMatrixType> xpetra_matrix_A =
        Teuchos::rcp(new XpetraCrsMatrixWrapType(xpetra_A));
      Teuchos::RCP<LinearOperatorType> belos_A =
        Teuchos::rcp(new XpetraOpType(xpetra_matrix_A));

      // Create a LinearProblem struct with the problem to solve.
      linear_problem = Teuchos::rcp(
        new Belos::LinearProblem<Number, MultiVectorType, LinearOperatorType>(
          belos_A, xpetra_x, xpetra_b));

      linear_problem->setProblem(xpetra_x, xpetra_b);
      do_solve(preconditioner);

      // Cast the result back to Vector
      Teuchos::RCP<XpetraTpetraMultiVectorType> xpetra_tpetra_x =
        Teuchos::rcp(new XpetraTpetraMultiVectorType(*xpetra_x));

      x = Vector(Teuchos::rcp(
        new VectorType(*(xpetra_tpetra_x->getTpetra_MultiVector()), 0)));
    }

#  ifndef DOXYGEN
#    include "trilinos_tpetra_solver.inst"
#  endif
  } // namespace TpetraWrappers
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_TPETRA
