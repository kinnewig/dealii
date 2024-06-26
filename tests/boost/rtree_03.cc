/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 */

// Check that we can construct a boost rtree based on indices.

#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/rtree.h>

#include <boost/geometry/algorithms/buffer.hpp>

#include "../tests.h"

namespace bgi = boost::geometry::index;

int
main()
{
  initlog();
  const unsigned int    N = 30;
  std::vector<Point<2>> points(N);
  for (auto &p : points)
    p = random_point<2>();

  auto tree  = pack_rtree(points);
  auto tree2 = pack_rtree_of_indices(points);

  Point<2>   p0(0, 0);
  Point<2>   p1(.4, .7111);
  Segment<2> segment(p0, p1);

  for (const auto &p : tree | bgi::adaptors::queried(bgi::nearest(segment, 3)))
    deallog << "P: " << p << std::endl;

  deallog << "Same with other tree?" << std::endl;

  for (const auto &i : tree2 | bgi::adaptors::queried(bgi::nearest(segment, 3)))
    deallog << "P[" << i << "]: " << points[i] << std::endl;
}
