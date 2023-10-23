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

#  include <deal.II/lac/trilinos_tpetra_precondition.h>
#  include <Ifpack2_Factory.hpp>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{

  namespace TpetraWrappers
  {
    // PreconditionBase
    template <typename Number, typename Node, typename LinearOperator, typename MultiVector>
    PreconditionBase<Number, Node, LinearOperator, MultiVector>::PreconditionBase(){
      // Make an empty new parameter list.
      parameter_list = Teuchos::parameterList();
    };



    template <typename Number, typename Node, typename LinearOperator, typename MultiVector>
    LinearOperator &
    PreconditionBase<Number, Node, LinearOperator, MultiVector>::trilinos_operator() const
    {
      AssertThrow(!preconditioner.is_null(),
                  ExcMessage("Trying to dereference a null pointer."));
      return (*preconditioner);
    }



    template <typename Number, typename Node, typename LinearOperator, typename MultiVector>
    Teuchos::RCP<LinearOperator>
    PreconditionBase<Number, Node, LinearOperator, MultiVector>::trilinos_rcp() const
    {
      AssertThrow(!preconditioner.is_null(),
                  ExcMessage("Trying to dereference a null pointer."));
      return (preconditioner);
    }



    // PreconditionILUT
    template <typename Number, typename Node>
    void PreconditionRILUK<Number, Node>::initialize(
      const SparseMatrix<double> &matrix)
    {
      // release memory before reallocation
      preconditioner.reset();


      // TODO: BEGIN Parameterlist:
      //       We set the parameter for the preconditioner here

      // Preconditioner to use
      preconditioner_type = "RILUK";

      // Make an empty new parameter list.
      parameter_list = Teuchos::parameterList();

      const double fillLevel = 2.0;
      const double dropTol = 0.0;
      const double absThreshold = 0.1;


      parameter_list->set ("fact: ilut level-of-fill", fillLevel);
      parameter_list->set ("fact: drop tolerance", dropTol);
      parameter_list->set ("fact: absolute threshold", absThreshold);
      // TODO: END ParameterList

      // Create the Ifpack2 preconditioner
      Teuchos::RCP<
        Ifpack2::Preconditioner<Number, int, long long int>>
        prec;

      Ifpack2::Factory factory;

      // Set up the preconditioner of the given type
      prec =
        factory.create(preconditioner_type, matrix.trilinos_rcp().getConst());

      prec->setParameters(*parameter_list);

      prec->initialize();

      // THIS ACTUALLY COMPUTES THE PRECONDITIONER
      prec->compute();

      preconditioner = prec;
    }



    // FROSch
    template <typename Number, typename Node>
    void
    PreconditionFROSch<Number, Node>::initialize(const SparseMatrix<Number> &matrix)
    {
      // TODO: ParameterList
      // Make an empty new parameter list.
      parameter_list = Teuchos::parameterList();

      // create a Xpetra::CrsMatrix Object,
      // which is used to create the Xpetra::Matrix
      Teuchos::RCP<XpetraCrsMatrixType> xpetra_crsmatrix
        = Teuchos::rcp(new XpetraTpetraCrsMatrixType(matrix.trilinos_rcp()));

      // Create from the above defined Xpetra::CrsMatrix
      // an Xpetra::Matrix
      Teuchos::RCP<XpetraMatrixType> xpetra_matrix
        = Teuchos::rcp(new XpetraCrsMatrixWrapType(xpetra_crsmatrix));

      // The one-level Schwarz preconditioner object
      Teuchos::RCP<OneLevelPreconditionerType> prec(
        new OneLevelPreconditionerType(xpetra_matrix, parameter_list));

      // Initialize
      prec->initialize(false);

      // THIS ACTUALLY COMPUTES THE PRECONDITIONER
      prec->compute();

      preconditioner = rcp(new XpetraOpType(prec));
    }

  } // namespace TpetraWrapper
}// namespace LinearAlgebra

namespace  LinearAlgebra
  {
  namespace TpetraWrappers
    {
      template class PreconditionBase<double>;
      template class PreconditionRILUK<double>;

      template class PreconditionBase<
        double,
        Tpetra::KokkosClassic::DefaultNode::DefaultNodeType,
        Belos::OperatorT<Xpetra::MultiVector<double, int, dealii::types::signed_global_dof_index>>,
        Xpetra::MultiVector<double, int, dealii::types::signed_global_dof_index>>;
      template class PreconditionFROSch<double>;
    } // namespace TpetraWrapper
  }// namespace LinearAlgebra


DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_TPETRA
