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



// test LinearAlgebra::TpetraWrappers::MatrixBase::const_iterator

#include <deal.II/base/utilities.h>

#include <deal.II/lac/trilinos_tpetra_sparse_matrix.h>

#include <iostream>

#include "../tests.h"


void
test()
{
  LinearAlgebra::TpetraWrappers::SparseMatrix<double, MemorySpace::Default> m(
    5U, 5U, 5U);
  m.set(0, 0, 1);
  m.set(1, 1, 2);
  m.set(1, 2, 3);
  m.compress(VectorOperation::insert);
  LinearAlgebra::TpetraWrappers::SparseMatrix<double, MemorySpace::Default>::
    const_iterator i = m.begin();
  deallog << i->row() << ' ' << i->column() << ' ' << i->value() << std::endl;
  ++i;
  deallog << i->row() << ' ' << i->column() << ' ' << i->value() << std::endl;
  i++;
  deallog << i->row() << ' ' << i->column() << ' ' << i->value() << std::endl;

  deallog << "OK" << std::endl;
}



int
main(int argc, char **argv)
{
  initlog();

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());


  try
    {
      {
        test();
      }
    }
  catch (const std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };
}
