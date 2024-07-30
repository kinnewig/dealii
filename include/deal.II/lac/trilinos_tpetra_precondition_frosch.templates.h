// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_trilinos_tpetra_precondition_frosch_templates_h
#define dealii_trilinos_tpetra_precondition_frosch_templates_h

#include <deal.II/base/config.h>

#include <deal.II/base/index_set.h>

#include <deal.II/lac/trilinos_tpetra_types.h>
#include <deal.II/lac/trilinos_xpetra_types.h>

#include <Teuchos_ParameterList.hpp>

#include <string>

#include "Teuchos_RCP.hpp"

#ifdef DEAL_II_TRILINOS_WITH_TPETRA

#  include <deal.II/lac/trilinos_tpetra_precondition.h>
#  include <deal.II/lac/trilinos_tpetra_precondition.templates.h>

// TODO Check DEAL_II_TRILINOS_WITH_FROSCH
// #  ifdef DEAL_II_TRILINOS_WITH_FROSCH
//     //FROSch
#  include <FROSch_OneLevelPreconditioner_def.hpp>
#  include <FROSch_SchwarzPreconditioners_fwd.hpp>
// TODO: Add here the Trilinos version, where this feature was merged
// #    if DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
#  include <FROSch_GeometricOneLevelPreconditioner_decl.hpp>
#  include <FROSch_GeometricOneLevelPreconditioner_def.hpp>
#  include <FROSch_GeometricTwoLevelPreconditioner_decl.hpp>
#  include <FROSch_GeometricTwoLevelPreconditioner_def.hpp>
// #    endif // DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
#  include <FROSch_Tools_def.hpp>
#  include <ShyLU_DDFROSch_config.h>
// #  endif // DEAL_II_TRILINOS_WITH_FROSCH

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace TpetraWrappers
  {

    // TODO Check DEAL_II_TRILINOS_WITH_FROSCH
    //   #ifdef DEAL_II_TRILINOS_WITH_FROSCH

    /* ---------------------- PreconditionFROSch ------------------------ */
    // Function to convert a Xpetra::Operator into a Tpetra::Operator
    namespace internal
    {
      template <typename Number, typename MemorySpace>
      Teuchos::RCP<TpetraTypes::LinearOperator<Number, MemorySpace>>
      XpetraToTpetra(Teuchos::RCP<XpetraTypes::LinearOperator<Number, MemorySpace>> xpetra_prec)
      {
        Teuchos::RCP<XpetraTypes::XpetraToTpetraLinearOperator<Number, MemorySpace>> 
          frosch_tpetra_prec = Teuchos::rcp(new XpetraTypes::XpetraToTpetraLinearOperator<Number, MemorySpace>(xpetra_prec));

        Teuchos::RCP<TpetraTypes::LinearOperator<Number, MemorySpace>>
          tpetra_prec = Teuchos::rcp_dynamic_cast<
            TpetraTypes::LinearOperator<Number, MemorySpace>>(
            frosch_tpetra_prec);

        return tpetra_prec;
      }
    } // namespace internal



    template <typename Number, typename MemorySpace>
    PreconditionFROSch<Number, MemorySpace>::PreconditionFROSch(
      const std::string &precondition_type)
      : precondition_type(precondition_type)
    {}


    template <typename Number, typename MemorySpace>
    void
    PreconditionFROSch<Number, MemorySpace>::initialize(
      const SparseMatrix<Number, MemorySpace> &A,
      Teuchos::RCP<Teuchos::ParameterList>    &parameters)
    {
      // store the parameter list
      this->parameter_list.setParameters(*parameters);

      // create a Xpetra::CrsMatrix Object,
      // which is used to create the Xpetra::Matrix
      Teuchos::RCP<XpetraTypes::CrsMatrixType<Number, MemorySpace>>
        xpetra_crsmatrix = Teuchos::rcp(
          new XpetraTypes::TpetraCrsMatrixType<Number, MemorySpace>(
            Teuchos::rcp_const_cast<
              TpetraTypes::MatrixType<Number, MemorySpace>>(A.trilinos_rcp())));

      // Create from the above defined Xpetra::CrsMatrix
      // an Xpetra::Matrix
      Teuchos::RCP<XpetraTypes::MatrixType<Number, MemorySpace>> xpetra_matrix =
        Teuchos::rcp(new XpetraTypes::CrsMatrixWrapType<Number, MemorySpace>(
          xpetra_crsmatrix));

      if (precondition_type == "one_level")
        {
          // The one-level Schwarz preconditioner object
          Teuchos::RCP<XpetraTypes::FROSchOneLevelType<Number, MemorySpace>>
            prec(new XpetraTypes::FROSchOneLevelType<Number, MemorySpace>(
              xpetra_matrix,
              Teuchos::rcpFromRef<Teuchos::ParameterList>(
                this->parameter_list)));

          // Initialize
          prec->initialize(false);

          // THIS ACTUALLY COMPUTES THE PRECONDITIONER
          prec->compute();

          // convert the FROSch preconditioner into a Xpetra::Operator
          Teuchos::RCP<XpetraTypes::LinearOperator<Number, MemorySpace>>
            xpetra_prec = Teuchos::rcp_dynamic_cast<
              XpetraTypes::LinearOperator<Number, MemorySpace>>(
              prec);

          // convert the FROSch preconditioner into a Tpetra::Operator
          // (The OneLevelOperator is derived from the Xpetra::Operator)
          this->preconditioner = internal::XpetraToTpetra<Number, MemorySpace>(xpetra_prec);
        }
      else if (precondition_type == "two_level")
        {
          // The two-level Schwarz preconditioner object
          Teuchos::RCP<XpetraTypes::FROSchTwoLevelType<Number, MemorySpace>>
            prec(new XpetraTypes::FROSchTwoLevelType<Number, MemorySpace>(
              xpetra_matrix,
              Teuchos::rcpFromRef<Teuchos::ParameterList>(
                this->parameter_list)));

          // Initialize
          prec->initialize(false);

          // THIS ACTUALLY COMPUTES THE PRECONDITIONER
          prec->compute();

          // convert the FROSch preconditioner into a Xpetra::Operator
          Teuchos::RCP<XpetraTypes::LinearOperator<Number, MemorySpace>>
            xpetra_prec = Teuchos::rcp_dynamic_cast<
              XpetraTypes::LinearOperator<Number, MemorySpace>>(
              prec);

          // convert the FROSch preconditioner into a Tpetra::Operator
          // (The OneLevelOperator is derived from the Xpetra::Operator)
          this->preconditioner = internal::XpetraToTpetra<Number, MemorySpace>(xpetra_prec);

        }
      else
        {
          AssertThrow(false, dealii::ExcNotImplemented());
        }
    }



    // TODO: Add here the Trilinos version, where this feature was merged
    // #  if DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
    /* ---------------- PreconditionGeometricFROSch ------------------- */
    template <typename Number, typename MemorySpace>
    PreconditionGeometricFROSch<Number, MemorySpace>::
      PreconditionGeometricFROSch(const std::string &precondition_type)
      : precondition_type(precondition_type)
    {}

    template <typename Number, typename MemorySpace>
    void
    PreconditionGeometricFROSch<Number, MemorySpace>::initialize(
      Teuchos::RCP<XpetraTypes::FROSchGeometricOneLevelType<Number, MemorySpace>>
        prec)
    {
      // convert the FROSch preconditioner into a Xpetra::Operator
      Teuchos::RCP<XpetraTypes::LinearOperator<Number, MemorySpace>>
        xpetra_prec = Teuchos::rcp_dynamic_cast<
          XpetraTypes::LinearOperator<Number, MemorySpace>>(
          prec);

      // convert the FROSch preconditioner into a Tpetra::Operator
      // (The OneLevelOperator is derived from the Xpetra::Operator)
      this->preconditioner = internal::XpetraToTpetra<Number, MemorySpace>(xpetra_prec);
    }

    template <typename Number, typename MemorySpace>
    void
    PreconditionGeometricFROSch<Number, MemorySpace>::initialize(
      Teuchos::RCP<XpetraTypes::FROSchGeometricTwoLevelType<Number, MemorySpace>>
        prec)
    {
      // convert the FROSch preconditioner into a Xpetra::Operator
      Teuchos::RCP<XpetraTypes::LinearOperator<Number, MemorySpace>>
        xpetra_prec = Teuchos::rcp_dynamic_cast<
          XpetraTypes::LinearOperator<Number, MemorySpace>>(
          prec);

      // convert the FROSch preconditioner into a Tpetra::Operator
      // (The OneLevelOperator is derived from the Xpetra::Operator)
      this->preconditioner = internal::XpetraToTpetra<Number, MemorySpace>(xpetra_prec);
    }

    // #  endif // DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)



    // TODO End check DEAL_II_TRILINOS_WITH_FROSCH
    // #  endif // DEAL_II_TRILINOS_WITH_FROSCH

  } // namespace TpetraWrappers
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_TPETRA

#endif // dealii_trilinos_tpetra_precondition_frosch_templates_h
