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


#include <deal.II/lac/trilinos_tpetra_precondition_frosch.templates.h>

#ifdef DEAL_II_TRILINOS_WITH_TPETRA
// TODO Check DEAL_II_TRILINOS_WITH_FROSCH
// #  ifdef DEAL_II_TRILINOS_WITH_FROSCH

DEAL_II_NAMESPACE_OPEN

#  ifndef DOXYGEN
// explicit instantiations
namespace LinearAlgebra
{
  namespace TpetraWrappers
  {
#    ifdef HAVE_TPETRA_INST_FLOAT
    template class PreconditionFROSch<float>;
#    endif

#    ifdef HAVE_TPETRA_INST_DOUBLE
    template class PreconditionFROSch<double>;
#    endif

#    ifdef DEAL_II_WITH_COMPLEX_VALUES
#      ifdef HAVE_TPETRA_INST_COMPLEX_FLOAT
    template class PreconditionFROSchu<std::complex<float>>;
#      endif
#      ifdef HAVE_TPETRA_INST_COMPLEX_DOUBLE
    template class PreconditionFROSch<std::complex<double>>;
#      endif
#    endif


// TODO: Add here the Trilinos version, where this feature was merged
// #        if DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)
#    ifdef HAVE_TPETRA_INST_FLOAT
    template class PreconditionGeometricFROSch<float>;
#    endif

#    ifdef HAVE_TPETRA_INST_DOUBLE
    template class PreconditionGeometricFROSch<double>;
#    endif

#    ifdef DEAL_II_WITH_COMPLEX_VALUES
#      ifdef HAVE_TPETRA_INST_COMPLEX_FLOAT
    template class PreconditionGeometricFROSchu<std::complex<float>>;
#      endif
#      ifdef HAVE_TPETRA_INST_COMPLEX_DOUBLE
    template class PreconditionGeometricFROSch<std::complex<double>>;
#      endif
#    endif
    // #        endif // DEAL_II_TRILINOS_VERSION_GTE(16, 0, 0)

  } // namespace TpetraWrappers

} // namespace LinearAlgebra

#  endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

// TODO End check DEAL_II_TRILINOS_WITH_FROSCH
// #  endif // DEAL_II_TRILINOS_WITH_FROSCH
#endif // DEAL_II_TRILINOS_WITH_TPETRA
