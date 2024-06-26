// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2006 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>

#include "../tests.h"


// This a test for the hp-capable version of the make_hanging_node_constraints
// method. It uses a triangulation with one refined element beside an
// unrefined element to create the constraints for this configuration.

template <int dim, int spacedim>
int
generate_grid(Triangulation<dim, spacedim> &tria)
{
  Point<dim>                p1, p2;
  std::vector<unsigned int> sub_div;

  // Define a rectangular shape
  for (unsigned int d = 0; d < dim; ++d)
    {
      p1[d] = 0;
      p2[d] = (d == 0) ? 2.0 : 1.0;
      sub_div.push_back((d == 0) ? 2 : 1);
    }
  GridGenerator::subdivided_hyper_rectangle(tria, sub_div, p1, p2, true);

  // Refine the first cell.
  tria.begin_active()->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  return (0);
}



template <int dim, int spacedim>
void
test_constraints(hp::FECollection<dim, spacedim> &fe_coll)
{
  Triangulation<dim, spacedim> tria;

  // Setup a rectangular domain
  // where one cell is h-refined,
  // while the other cell is
  // unrefined. Furthermore every cell
  // gets a different active_fe_index.
  // This should serve as a testcase
  // for the hanging node constraints.
  generate_grid(tria);

  // Now assign increasing
  // active_fe_indices to
  // the different cells.
  DoFHandler<dim, spacedim> dof_handler(tria);
  typename DoFHandler<dim, spacedim>::active_cell_iterator
    cell               = dof_handler.begin_active(),
    endc               = dof_handler.end();
  unsigned int fe_indx = 0;
  for (; cell != endc; ++cell)
    {
      cell->set_active_fe_index(fe_indx);
      ++fe_indx;
    }

  // Distribute DoFs;
  dof_handler.distribute_dofs(fe_coll);
  deallog << "DoFs: " << dof_handler.n_dofs() << std::endl;

  // Create the constraints.
  AffineConstraints<double> constraint_matrix;

  DoFTools::make_hanging_node_constraints(dof_handler, constraint_matrix);

  // Output the constraints
  constraint_matrix.print(deallog.get_file_stream());
}


template <int dim, int spacedim>
void
test_constraints_old(FiniteElement<dim, spacedim> &fe)
{
  Triangulation<dim, spacedim> tria;

  // Setup a rectangular domain
  // where one cell is h-refined,
  // while the other cell is
  // unrefined. Furthermore every cell
  // gets a different active_fe_index.
  // This should serve as a testcase
  // for the hanging node constraints.
  generate_grid(tria);

  // Now assign increasing
  // active_fe_indices to
  // the different cells.
  DoFHandler<dim, spacedim> dof_handler(tria);

  // Distribute DoFs;
  dof_handler.distribute_dofs(fe);
  deallog << "DoFs: " << dof_handler.n_dofs() << std::endl;

  // Create the constraints.
  AffineConstraints<double> constraint_matrix;

  DoFTools::make_hanging_node_constraints(dof_handler, constraint_matrix);

  // Output the constraints
  constraint_matrix.print(deallog.get_file_stream());
}

template <int dim, int spacedim>
void
check()
{
  FE_Q<dim, spacedim> fe_1(1);
  FE_Q<dim, spacedim> fe_2(2);
  FE_Q<dim, spacedim> fe_3(QIterated<1>(QTrapezoid<1>(), 3));

  hp::FECollection<dim, spacedim> fe_coll2;
  fe_coll2.push_back(fe_3);
  fe_coll2.push_back(fe_2);
  fe_coll2.push_back(fe_2);

  fe_coll2.push_back(fe_2);
  fe_coll2.push_back(fe_3);

  test_constraints<dim, spacedim>(fe_coll2);

  test_constraints_old<dim, spacedim>(fe_1);
}



int
main()
{
  initlog();
  deallog.get_file_stream().precision(2);

  check<2, 2>();
  check<2, 3>();

  return 0;
}
