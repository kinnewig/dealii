// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2012 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// tests for the BlockMask class
//
// here: BlockMask::operator==


#include <deal.II/fe/block_mask.h>

#include "../tests.h"



void
test()
{
  std::vector<bool> v1(12);
  for (unsigned int i = 0; i < v1.size(); ++i)
    v1[i] = (i % 3 == 0);
  std::vector<bool> v2(12);
  for (unsigned int i = 0; i < v2.size(); ++i)
    v2[i] = (i % 4 == 0);

  std::vector<bool> v(12);
  for (unsigned int i = 0; i < v.size(); ++i)
    v[i] = (v1[i] || v2[i]);

  BlockMask m1(v1);
  BlockMask m2(v2);
  BlockMask m = m1 | m2;

  // verify equality
  AssertThrow(m == BlockMask(v), ExcInternalError());
  AssertThrow(!(m == m1), ExcInternalError());
  AssertThrow(!(m == BlockMask(v1)), ExcInternalError());
  AssertThrow(!(m == BlockMask(v2)), ExcInternalError());

  deallog << "OK" << std::endl;
}


int
main()
{
  initlog();
  deallog << std::setprecision(4);

  test();
}
