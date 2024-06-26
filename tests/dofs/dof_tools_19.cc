// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2006 - 2020 by the deal.II authors
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

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include "dof_tools_common.h"

// check
//   DoFTools::
//   make_hanging_node_constraints (const DoFHandler<dim> &,
//                              AffineConstraints<double>      &);
//
// As the results of dof_tools_03 seem unclear, this test aims
// at verifying the constraints by projecting a constant function
// onto the FE space and afterwards evaluating the L_2 error.
// This should reveal errors in the constraint matrix.

// as a side effect, this test used to crash because a)
// VectorTools::project instantiated QGauss<0>(6) when in 1d, and b)
// because VectorTools::project wasn't implemented at all in 1d. It
// required fixing both bugs to get to the actual point of this test.



template <int dim>
void
check_this(const DoFHandler<dim> &dof_handler)
{
  // there's presently a crash in the
  // Raviart-Thomas element. don't
  // check this element for that
  // reason. this case is covered in
  // rt_crash_01, however
  if (dof_handler.get_fe().get_name().find("RaviartThomas") !=
      std::string::npos)
    return;

  Functions::ConstantFunction<dim> test_func(
    1, dof_handler.get_fe().n_components());

  // don't run this test if hanging
  // nodes are not implemented
  if (dof_handler.get_fe().constraints_are_implemented() == false)
    return;

  AffineConstraints<double> cm;
  DoFTools::make_hanging_node_constraints(dof_handler, cm);
  cm.close();

  deallog << cm.n_constraints() << std::endl;
  deallog << cm.max_constraint_indirections() << std::endl;

  // L_2 project constant function onto field
  QGauss<dim>    quadrature(6);
  Vector<double> solution(dof_handler.n_dofs());

  VectorTools::project(dof_handler, cm, quadrature, test_func, solution);
  cm.distribute(solution);

  // Evaluate error
  Vector<double> cellwise_errors(
    dof_handler.get_triangulation().n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    test_func,
                                    cellwise_errors,
                                    quadrature,
                                    VectorTools::L2_norm);
  const double p_l2_error = cellwise_errors.l2_norm();

  Assert(p_l2_error < 1e-11, ExcInternalError());

  deallog << "L2_Error : " << p_l2_error << std::endl;
}
