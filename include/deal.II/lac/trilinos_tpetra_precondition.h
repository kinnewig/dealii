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

#ifndef dealii_trilinos_belos_precondition_h
#define dealii_trilinos_belos_precondition_h

#include <deal.II/base/config.h>

#  ifdef DEAL_II_TRILINOS_WITH_TPETRA

#  include <deal.II/base/subscriptor.h>
#  include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>

#  include <Teuchos_ParameterList.hpp>
#  include <Tpetra_Operator.hpp>

// Belos
#include <BelosLinearProblem.hpp>
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosPseudoBlockGmresSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>

// Galeri::Xpetra
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"
#include "Galeri_XpetraParameters.hpp"
#include "Galeri_XpetraUtils.hpp"
#include "Galeri_XpetraMaps.hpp"

// Teuchos
#include <Teuchos_Array.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_Tuple.hpp>

// Thyra
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_SolveSupportTypes.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorStdOps.hpp>
#ifdef HAVE_SHYLU_DDFROscalar_typeH_EPETRA
#include <Thyra_EpetraLinearOp.hpp>
#endif
#include <Thyra_VectorSpaceBase_def.hpp>
#include <Thyra_VectorSpaceBase_decl.hpp>

// Xpetra
#include <Xpetra_Map.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <Xpetra_Parameters.hpp>

// TODO Check DEAL_II_WITH_FROSCH
// FROSch
#include <ShyLU_DDFROSch_config.h>
#include <FROSch_Tools_def.hpp>
#include <FROSch_SchwarzPreconditioners_fwd.hpp>
#include <FROSch_OneLevelPreconditioner_def.hpp>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup TpetraWrappers
 * @{
 */
namespace LinearAlgebra
{
  namespace TpetraWrappers
  {
    // forward declarations
#  ifndef DOXYGEN
    template <typename Number>
    class SparseMatrix;

    template <typename Number>
    class Vector;

    class SparsityPattern;

    template <typename Number, typename Node, typename LinearOperator, typename MultiVector>
    class SolverBase;
#  endif

    /**
     * The base class for all preconditioners based on Trilinos Tpetra sparse matrices.
     *
     * @ingroup TpetraWrappers
     * @ingroup Preconditioners
     */
    template <typename Number,
              typename Node           = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType,
              typename LinearOperator = Tpetra::Operator<Number, int, dealii::types::signed_global_dof_index, Node>,
              typename MultiVector    = Tpetra::MultiVector<Number, int, dealii::types::signed_global_dof_index, Node>>
    class PreconditionBase : public Subscriptor
    {
    public:
      /**
       * Declare the valye type
       */
      using value_type = Number;

      /**
       * Declare the type for container size.
       */
      using size_type = dealii::types::global_dof_index;

      /*
       * Teuchos::ParameterList to pipe additional flags to the
       * preconditioner.
       */
      Teuchos::RCP<Teuchos::ParameterList> parameter_list;

      /**
       * Preconditioner Tyoe
       */
      std::string preconditioner_type;

      /**
       * Constructor. Does not do anything. The <tt>initialize</tt> function of
       * the derived classes will have to create the preconditioner from a given
       * sparse matrix.
       */
      PreconditionBase();

      /**
       * @name Access to underlying Trilinos data
       */
      /** @{ */
      /**
       *
       * Calling this function from an uninitialized object will cause an
       * exception.
       */
       // TODO: is it a good idea to return the object itself instead of a RCP?
      LinearOperator &
      trilinos_operator() const;

      /**
       * Return a Teuchos::RCP to the underlying Tpetra::Operator
       */
       Teuchos::RCP<LinearOperator>
       trilinos_rcp() const;
      /** @} */


      /**
     * @addtogroup Exceptions
       */
      /** @{ */
      /**
     * Exception.
       */
      DeclException1(ExcNonMatchingMaps,
                     std::string,
                     << "The sparse matrix the preconditioner is based on "
                       << "uses a map that is not compatible to the one in vector "
                       << arg1 << ". Check preconditioner and matrix setup.");
      /** @} */

      friend class SolverBase<Number, Node, LinearOperator, MultiVector>;

    protected:
      /**
       * This is a RCP to the preconditioner object that is used when
       * applying the preconditioner.
       */
      Teuchos::RCP<LinearOperator> preconditioner;

    }; // class PreconditionBase
       
      
     
    /**
    * TODO: Description
    */
    template <typename Number, typename Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class PreconditionRILUK : public PreconditionBase<Number, Node>
    {
      public:
        using PreconditionBase<Number, Node>::preconditioner_type;
        using PreconditionBase<Number, Node>::parameter_list;

        using PreconditionBase<Number, Node>::preconditioner;
        /**
         * Take the sparse matrix the preconditioner object should be built of,
         * and additional flags (damping parameter, overlap in parallel
         * computations, etc.) if there are any.
         */
        void
        initialize(const SparseMatrix<double> &matrix);
    }; // class PreconditionIULT



    /**
     * TODO: Description
     */
    template <typename Number, typename Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class PreconditionFROSch : public PreconditionBase<
                                        Number,
                                        Node,
                                        Belos::OperatorT<Xpetra::MultiVector<Number, int, dealii::types::signed_global_dof_index, Node>>,
                                        Xpetra::MultiVector<Number, int, dealii::types::signed_global_dof_index, Node>>
    {
     public:

      /**
       * Shorthand for the OneLevelPrecondioner of FROSch
       */
      using OneLevelPreconditionerType =
         FROSch::OneLevelPreconditioner<Number, int, dealii::types::signed_global_dof_index, Node>;
         //FROSch::TwoLevelPreconditioner<Number, int, dealii::types::signed_global_dof_index, Node>;

      using MultiVectorType =
        Xpetra::MultiVector<Number, int, dealii::types::signed_global_dof_index, Node>;

      using LinearOperatorType =
        Belos::OperatorT<MultiVectorType>;

      using XpetraOpType =
        Belos::XpetraOp<Number, int, dealii::types::signed_global_dof_index, Node>;

      using XpetraMatrixType =
        Xpetra::Matrix<Number, int, dealii::types::signed_global_dof_index, Node>;

      using XpetraCrsMatrixType =
        Xpetra::CrsMatrix<Number, int, dealii::types::signed_global_dof_index, Node>;

      using XpetraTpetraCrsMatrixType =
        Xpetra::TpetraCrsMatrix<Number, int, dealii::types::signed_global_dof_index, Node>;

      using XpetraCrsMatrixWrapType =
        Xpetra::CrsMatrixWrap<Number, int, dealii::types::signed_global_dof_index, Node>;


      using PreconditionBase<Number, Node, LinearOperatorType, MultiVectorType>::preconditioner_type;
      using PreconditionBase<Number, Node, LinearOperatorType, MultiVectorType>::parameter_list;
      using PreconditionBase<Number, Node, LinearOperatorType, MultiVectorType>::preconditioner;

      void initialize(const SparseMatrix<Number> &matrix);
    };


  } // namespace TpetraWrapper
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#  endif // DEAL_II_TRILINOS_WITH_TPETRA

#endif // dealii_trilinos_tpetra_precondition_h
