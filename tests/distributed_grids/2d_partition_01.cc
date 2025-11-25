/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2025 - 2025 Lukas Dreyer
 *
 * The code is licensed under the GNU Lesser General Public License as 
 * published by the Free Software Foundation in version 2.1 
 * The full text of the license can be found in the file LICENSE.md
 *
 * ---------------------------------------------------------------------
 * Contact:
 *   Lukas Dreyer
 *   Leibniz Universität Hannover (LUH)
 *   Institut für Angewandte Mathematik (IfAM)
 *
 * Questions?
 *   E-Mail: dreyeyr@ifam.uni-hannover.de
 *
 * Date: Nov 21, 2025
 *
 * ---------------------------------------------------------------------
 *
 * Check that the elements of a grid are partitioned evenly
 *  
 */


#include "../tests.h"

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/base/conditional_ostream.h>


#include "deal.II/grid/reference_cell.h"
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <iostream>
#include <fstream>

namespace MinimalExample
{
  using namespace dealii;

  template <int dim>
  class EmptyRank
  {
  public:
    EmptyRank();

    void
    run();

  private:
    void
    make_grid();

    void
    communicate_and_print_empty();

    // === Member ===
    // MPI communicator
    MPI_Comm mpi_communicator;

    // Parallel distributed triangulation
    parallel::distributed::Triangulation<dim> triangulation;
  };


  template <int dim>
  EmptyRank<dim>::EmptyRank()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
  {}



  template <int dim>
  void
  EmptyRank<dim>::make_grid()
  {
    GridGenerator::reference_cell(triangulation, ReferenceCells::Quadrilateral);
    const unsigned int n_refinements = 4;
    triangulation.refine_global(n_refinements);

    // For debugging: 
    std::string name = "Grid.vtk";
    std::ofstream output_file(name);
    GridOut().write_vtk(triangulation, output_file);
  }



  template <int dim>
  void
  EmptyRank<dim>::communicate_and_print_empty()
  {
    // !!!ISSUE!!!
    // On some ranks there are no locally owned cells, we need to identfiy those
    // ranks. 
    
    // We begin by identfiying all ranks that do not own any cells.
    // Ranks that do own cells are flaged by "-1".
    int mpirank, mpisize;
    MPI_Comm_rank(mpi_communicator, &mpirank);
    MPI_Comm_size(mpi_communicator, &mpisize);

    unsigned int n_active_cells = triangulation.n_locally_owned_active_cells();
    std::vector<unsigned int> n_active_proc(mpirank==0?mpisize:0);

    MPI_Gather(&n_active_cells, 1, MPI_UNSIGNED,
               n_active_proc.data(), 1, MPI_UNSIGNED, 0, mpi_communicator);
    if (mpirank==0){
        for (unsigned int i=0; i < n_active_proc.size();i++){
            deallog<<i<<" "<<n_active_proc[i]<<std::endl;
        }
    }
  }



  template <int dim>
  void
  EmptyRank<dim>::run()
  {
    // create the grid
    make_grid();

    communicate_and_print_empty();
  }
} // namespace MinimalExample



int
main(int argc, char *argv[])
{
  initlog();
  try
    {
      using namespace dealii;
      using namespace MinimalExample;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        EmptyRank<2> empty_rank_problem;
        empty_rank_problem.run();


    }
  catch (std::exception &exc)
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
    }

  return 0;
}
