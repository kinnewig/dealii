// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2014 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// Test by Michal Wichrowski: ensure that
// DoFRenumbering::Cuthill_McKee also works on empty processors

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_iterator.h>

#include "../tests.h"

template <int dim>
void
test()
{
  parallel::distributed::Triangulation<dim> tr(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tr, -1.0, 1.0);
  DoFHandler<dim>        dofh(tr);
  static const FE_Q<dim> fe(1);
  dofh.distribute_dofs(fe);
  DoFRenumbering::Cuthill_McKee(dofh);
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  deallog.push(Utilities::int_to_string(myid));
  if (myid == 0)
    {
      initlog();

      deallog.push("2d");
      test<2>();
      deallog << "OK" << std::endl;
      deallog.pop();

      deallog.push("3d");
      test<3>();
      deallog << "OK" << std::endl;
      deallog.pop();
    }
  else
    {
      test<2>();
      test<3>();
    }
}
