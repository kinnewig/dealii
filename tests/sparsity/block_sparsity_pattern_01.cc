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



// Test reinit(BlockIndices...)


#include <deal.II/lac/block_sparsity_pattern.h>

#include "../tests.h"


int
main()
{
  initlog();
  deallog << std::setprecision(2) << std::fixed;

  BlockSparsityPattern                 sparsity;
  std::vector<types::global_dof_index> row_blocks(4);
  row_blocks[0] = 4;
  row_blocks[1] = 5;
  row_blocks[2] = 1;
  row_blocks[3] = 4;
  std::vector<types::global_dof_index> col_blocks(3);
  col_blocks[0] = 2;
  col_blocks[1] = 3;
  col_blocks[2] = 2;
  BlockIndices rows(row_blocks);
  BlockIndices cols(col_blocks);

  std::vector<std::vector<unsigned int>> row_length(
    cols.size(), std::vector<unsigned int>(rows.total_size()));
  for (unsigned int jb = 0; jb < row_length.size(); ++jb)
    for (unsigned int i = 0; i < row_length[jb].size(); ++i)
      {
        const unsigned int d = col_blocks[jb] - 1;
        row_length[jb][i]    = (i + 1) % d + 1;
      }

  for (unsigned int j = 0; j < row_length.size(); ++j)
    {
      for (unsigned int i = 0; i < row_length[j].size(); ++i)
        deallog << ' ' << row_length[j][i];
      deallog << std::endl;
    }

  sparsity.reinit(rows, cols, row_length);

  for (unsigned int ib = 0; ib < rows.size(); ++ib)
    for (unsigned int i = 0; i < rows.block_size(ib); ++i)
      {
        const unsigned int ii = rows.local_to_global(ib, i);
        for (unsigned int jb = 0; jb < cols.size(); ++jb)
          for (unsigned int j = 0; j < row_length[jb][ii]; ++j)
            sparsity.add(ii, cols.local_to_global(jb, j));
      }

  sparsity.print(deallog.get_file_stream());

  return 0;
}
