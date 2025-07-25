// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2000 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out_rotation.h>

#include <cmath>

DEAL_II_NAMESPACE_OPEN


// TODO: Update documentation
// TODO: Unify code for dimensions


// TODO: build_some_patches isn't going to work if first_cell/next_cell
// don't iterate over all cells and if cell data is requested. in that
// case, we need to calculate cell_number as in the DataOut class

// Not implemented for 3d


namespace internal
{
  namespace DataOutRotationImplementation
  {
    template <int dim, int spacedim>
    ParallelData<dim, spacedim>::ParallelData(
      const unsigned int               n_datasets,
      const unsigned int               n_subdivisions,
      const unsigned int               n_patches_per_circle,
      const std::vector<unsigned int> &n_postprocessor_outputs,
      const Mapping<dim, spacedim>    &mapping,
      const std::vector<
        std::shared_ptr<dealii::hp::FECollection<dim, spacedim>>>
                       &finite_elements,
      const UpdateFlags update_flags)
      : internal::DataOutImplementation::ParallelDataBase<dim, spacedim>(
          n_datasets,
          n_subdivisions,
          n_postprocessor_outputs,
          mapping,
          finite_elements,
          update_flags,
          false)
      , n_patches_per_circle(n_patches_per_circle)
    {}



    /**
     * In a WorkStream context, use this function to append the patch computed
     * by the parallel stage to the array of patches.
     */
    template <int dim, int spacedim>
    void
    append_patch_to_list(
      const std::vector<
        DataOutBase::Patch<DataOutRotation<dim, spacedim>::patch_dim,
                           DataOutRotation<dim, spacedim>::patch_spacedim>>
        &new_patches,
      std::vector<
        DataOutBase::Patch<DataOutRotation<dim, spacedim>::patch_dim,
                           DataOutRotation<dim, spacedim>::patch_spacedim>>
        &patches)
    {
      for (unsigned int i = 0; i < new_patches.size(); ++i)
        {
          patches.push_back(new_patches[i]);
          patches.back().patch_index = patches.size() - 1;
        }
    }
  } // namespace DataOutRotationImplementation
} // namespace internal



template <int dim, int spacedim>
void
DataOutRotation<dim, spacedim>::build_one_patch(
  const cell_iterator                                                  *cell,
  internal::DataOutRotationImplementation::ParallelData<dim, spacedim> &data,
  std::vector<DataOutBase::Patch<patch_dim, patch_spacedim>> &my_patches)
{
  if (dim == 3)
    {
      // would this function make any sense after all? who would want to
      // output/compute in four space dimensions?
      DEAL_II_NOT_IMPLEMENTED();
      return;
    }

  Assert((*cell)->is_locally_owned(), ExcNotImplemented());

  const unsigned int n_patches_per_circle = data.n_patches_per_circle;

  // another abbreviation denoting the number of q_points in each direction
  const unsigned int n_points = data.n_subdivisions + 1;

  // set up an array that holds the directions in the plane of rotation in
  // which we will put points in the whole domain (not the rotationally
  // reduced one in which the computation took place. for simplicity add the
  // initial direction at the end again
  std::vector<Point<dim + 1>> angle_directions(n_patches_per_circle + 1);
  for (unsigned int i = 0; i <= n_patches_per_circle; ++i)
    {
      angle_directions[i][dim - 1] =
        std::cos(2 * numbers::PI * i / n_patches_per_circle);
      angle_directions[i][dim] =
        std::sin(2 * numbers::PI * i / n_patches_per_circle);
    }

  for (unsigned int angle = 0; angle < n_patches_per_circle; ++angle)
    {
      // first compute the vertices of the patch. note that they will have to
      // be computed from the vertices of the cell, which has one dim
      // less, however.
      switch (dim)
        {
          case 1:
            {
              const double r1 = (*cell)->vertex(0)[0],
                           r2 = (*cell)->vertex(1)[0];
              Assert(r1 >= 0, ExcRadialVariableHasNegativeValues(r1));
              Assert(r2 >= 0, ExcRadialVariableHasNegativeValues(r2));

              my_patches[angle].vertices[0] = r1 * angle_directions[angle];
              my_patches[angle].vertices[1] = r2 * angle_directions[angle];
              my_patches[angle].vertices[2] = r1 * angle_directions[angle + 1];
              my_patches[angle].vertices[3] = r2 * angle_directions[angle + 1];

              break;
            }

          case 2:
            {
              for (const unsigned int vertex :
                   GeometryInfo<dim>::vertex_indices())
                {
                  const Point<dim> v = (*cell)->vertex(vertex);

                  // make sure that the radial variable is nonnegative
                  Assert(v[0] >= 0, ExcRadialVariableHasNegativeValues(v[0]));

                  // now set the vertices of the patch
                  my_patches[angle].vertices[vertex] =
                    v[0] * angle_directions[angle];
                  my_patches[angle].vertices[vertex][0] = v[1];

                  my_patches[angle]
                    .vertices[vertex + GeometryInfo<dim>::vertices_per_cell] =
                    v[0] * angle_directions[angle + 1];
                  my_patches[angle]
                    .vertices[vertex + GeometryInfo<dim>::vertices_per_cell]
                             [0] = v[1];
                }

              break;
            }

          default:
            DEAL_II_NOT_IMPLEMENTED();
        }

      // then fill in data
      if (data.n_datasets > 0)
        {
          unsigned int offset = 0;

          data.reinit_all_fe_values(this->dof_data, *cell);
          // first fill dof_data
          for (unsigned int dataset = 0; dataset < this->dof_data.size();
               ++dataset)
            {
              const FEValuesBase<dim> &fe_patch_values =
                data.get_present_fe_values(dataset);
              const unsigned int n_components =
                fe_patch_values.get_fe().n_components();
              const DataPostprocessor<dim> *postprocessor =
                this->dof_data[dataset]->postprocessor;
              if (postprocessor != nullptr)
                {
                  // we have to postprocess the
                  // data, so determine, which
                  // fields have to be updated
                  const UpdateFlags update_flags =
                    postprocessor->get_needed_update_flags();

                  if (n_components == 1)
                    {
                      // at each point there is
                      // only one component of
                      // value, gradient etc.
                      if (update_flags & update_values)
                        this->dof_data[dataset]->get_function_values(
                          fe_patch_values,
                          internal::DataOutImplementation::ComponentExtractor::
                            real_part,
                          data.patch_values_scalar.solution_values);
                      if (update_flags & update_gradients)
                        this->dof_data[dataset]->get_function_gradients(
                          fe_patch_values,
                          internal::DataOutImplementation::ComponentExtractor::
                            real_part,
                          data.patch_values_scalar.solution_gradients);
                      if (update_flags & update_hessians)
                        this->dof_data[dataset]->get_function_hessians(
                          fe_patch_values,
                          internal::DataOutImplementation::ComponentExtractor::
                            real_part,
                          data.patch_values_scalar.solution_hessians);

                      if (update_flags & update_quadrature_points)
                        data.patch_values_scalar.evaluation_points =
                          fe_patch_values.get_quadrature_points();

                      const typename DoFHandler<dim,
                                                spacedim>::active_cell_iterator
                        dh_cell(&(*cell)->get_triangulation(),
                                (*cell)->level(),
                                (*cell)->index(),
                                this->dof_data[dataset]->dof_handler);
                      data.patch_values_scalar.template set_cell<dim>(dh_cell);

                      postprocessor->evaluate_scalar_field(
                        data.patch_values_scalar,
                        data.postprocessed_values[dataset]);
                    }
                  else
                    {
                      data.resize_system_vectors(n_components);

                      // at each point there is a vector valued function and
                      // its derivative...
                      if (update_flags & update_values)
                        this->dof_data[dataset]->get_function_values(
                          fe_patch_values,
                          internal::DataOutImplementation::ComponentExtractor::
                            real_part,
                          data.patch_values_system.solution_values);
                      if (update_flags & update_gradients)
                        this->dof_data[dataset]->get_function_gradients(
                          fe_patch_values,
                          internal::DataOutImplementation::ComponentExtractor::
                            real_part,
                          data.patch_values_system.solution_gradients);
                      if (update_flags & update_hessians)
                        this->dof_data[dataset]->get_function_hessians(
                          fe_patch_values,
                          internal::DataOutImplementation::ComponentExtractor::
                            real_part,
                          data.patch_values_system.solution_hessians);

                      if (update_flags & update_quadrature_points)
                        data.patch_values_system.evaluation_points =
                          fe_patch_values.get_quadrature_points();

                      const typename DoFHandler<dim,
                                                spacedim>::active_cell_iterator
                        dh_cell(&(*cell)->get_triangulation(),
                                (*cell)->level(),
                                (*cell)->index(),
                                this->dof_data[dataset]->dof_handler);
                      data.patch_values_system.template set_cell<dim>(dh_cell);

                      postprocessor->evaluate_vector_field(
                        data.patch_values_system,
                        data.postprocessed_values[dataset]);
                    }

                  for (unsigned int component = 0;
                       component < this->dof_data[dataset]->n_output_variables;
                       ++component)
                    {
                      switch (dim)
                        {
                          case 1:
                            for (unsigned int x = 0; x < n_points; ++x)
                              for (unsigned int y = 0; y < n_points; ++y)
                                my_patches[angle].data(offset + component,
                                                       x * n_points + y) =
                                  data.postprocessed_values[dataset][x](
                                    component);
                            break;

                          case 2:
                            for (unsigned int x = 0; x < n_points; ++x)
                              for (unsigned int y = 0; y < n_points; ++y)
                                for (unsigned int z = 0; z < n_points; ++z)
                                  my_patches[angle].data(offset + component,
                                                         x * n_points *
                                                             n_points +
                                                           y * n_points + z) =
                                    data.postprocessed_values[dataset]
                                                             [x * n_points + z](
                                                               component);
                            break;

                          default:
                            DEAL_II_NOT_IMPLEMENTED();
                        }
                    }
                }
              else if (n_components == 1)
                {
                  this->dof_data[dataset]->get_function_values(
                    fe_patch_values,
                    internal::DataOutImplementation::ComponentExtractor::
                      real_part,
                    data.patch_values_scalar.solution_values);

                  switch (dim)
                    {
                      case 1:
                        for (unsigned int x = 0; x < n_points; ++x)
                          for (unsigned int y = 0; y < n_points; ++y)
                            my_patches[angle].data(offset, x * n_points + y) =
                              data.patch_values_scalar.solution_values[x];
                        break;

                      case 2:
                        for (unsigned int x = 0; x < n_points; ++x)
                          for (unsigned int y = 0; y < n_points; ++y)
                            for (unsigned int z = 0; z < n_points; ++z)
                              my_patches[angle].data(offset,
                                                     x * n_points * n_points +
                                                       y + z * n_points) =
                                data.patch_values_scalar
                                  .solution_values[x * n_points + z];
                        break;

                      default:
                        DEAL_II_NOT_IMPLEMENTED();
                    }
                }
              else
                // system of components
                {
                  data.resize_system_vectors(n_components);
                  this->dof_data[dataset]->get_function_values(
                    fe_patch_values,
                    internal::DataOutImplementation::ComponentExtractor::
                      real_part,
                    data.patch_values_system.solution_values);

                  for (unsigned int component = 0; component < n_components;
                       ++component)
                    {
                      switch (dim)
                        {
                          case 1:
                            for (unsigned int x = 0; x < n_points; ++x)
                              for (unsigned int y = 0; y < n_points; ++y)
                                my_patches[angle].data(offset + component,
                                                       x * n_points + y) =
                                  data.patch_values_system.solution_values[x](
                                    component);
                            break;

                          case 2:
                            for (unsigned int x = 0; x < n_points; ++x)
                              for (unsigned int y = 0; y < n_points; ++y)
                                for (unsigned int z = 0; z < n_points; ++z)
                                  my_patches[angle].data(offset + component,
                                                         x * n_points *
                                                             n_points +
                                                           y * n_points + z) =
                                    data.patch_values_system
                                      .solution_values[x * n_points + z](
                                        component);
                            break;

                          default:
                            DEAL_II_NOT_IMPLEMENTED();
                        }
                    }
                }
              offset += this->dof_data[dataset]->n_output_variables;
            }

          // then do the cell data
          for (unsigned int dataset = 0; dataset < this->cell_data.size();
               ++dataset)
            {
              // we need to get at the number of the cell to which this face
              // belongs in order to access the cell data. this is not readily
              // available, so choose the following rather inefficient way:
              Assert((*cell)->is_active(),
                     ExcMessage("Cell must be active for cell data"));
              const unsigned int cell_number = std::distance(
                this->triangulation->begin_active(),
                typename Triangulation<dim, spacedim>::active_cell_iterator(
                  *cell));
              const double value =
                this->cell_data[dataset]->get_cell_data_value(
                  cell_number,
                  internal::DataOutImplementation::ComponentExtractor::
                    real_part);
              switch (dim)
                {
                  case 1:
                    for (unsigned int x = 0; x < n_points; ++x)
                      for (unsigned int y = 0; y < n_points; ++y)
                        my_patches[angle].data(dataset + offset,
                                               x * n_points + y) = value;
                    break;

                  case 2:
                    for (unsigned int x = 0; x < n_points; ++x)
                      for (unsigned int y = 0; y < n_points; ++y)
                        for (unsigned int z = 0; z < n_points; ++z)
                          my_patches[angle].data(dataset + offset,
                                                 x * n_points * n_points +
                                                   y * n_points + z) = value;
                    break;

                  default:
                    DEAL_II_NOT_IMPLEMENTED();
                }
            }
        }
    }
}



template <int dim, int spacedim>
void
DataOutRotation<dim, spacedim>::build_patches(
  const unsigned int n_patches_per_circle,
  const unsigned int nnnn_subdivisions)
{
  Assert(this->triangulation != nullptr,
         Exceptions::DataOutImplementation::ExcNoTriangulationSelected());

  const unsigned int n_subdivisions =
    (nnnn_subdivisions != 0) ? nnnn_subdivisions : this->default_subdivisions;
  Assert(n_subdivisions >= 1,
         Exceptions::DataOutImplementation::ExcInvalidNumberOfSubdivisions(
           n_subdivisions));

  this->validate_dataset_names();

  unsigned int n_datasets = this->cell_data.size();
  for (unsigned int i = 0; i < this->dof_data.size(); ++i)
    n_datasets += this->dof_data[i]->n_output_variables;

  UpdateFlags update_flags = update_values | update_quadrature_points;
  for (unsigned int i = 0; i < this->dof_data.size(); ++i)
    if (this->dof_data[i]->postprocessor)
      update_flags |=
        this->dof_data[i]->postprocessor->get_needed_update_flags();
  // perhaps update_normal_vectors is present,
  // which would only be useful on faces, but
  // we may not use it here.
  Assert(!(update_flags & update_normal_vectors),
         ExcMessage("The update of normal vectors may not be requested for "
                    "evaluation of data on cells via DataPostprocessor."));

  // first count the cells we want to
  // create patches of and make sure
  // there is enough memory for that
  std::vector<cell_iterator> all_cells;
  for (cell_iterator cell = first_cell(); cell != this->triangulation->end();
       cell               = next_cell(cell))
    all_cells.push_back(cell);

  // then also take into account that
  // we want more than one patch to
  // come out of every cell, as they
  // are repeated around the axis of
  // rotation
  this->patches.clear();
  this->patches.reserve(all_cells.size() * n_patches_per_circle);


  std::vector<unsigned int> n_postprocessor_outputs(this->dof_data.size());
  for (unsigned int dataset = 0; dataset < this->dof_data.size(); ++dataset)
    if (this->dof_data[dataset]->postprocessor)
      n_postprocessor_outputs[dataset] =
        this->dof_data[dataset]->n_output_variables;
    else
      n_postprocessor_outputs[dataset] = 0;

  Assert(!this->triangulation->is_mixed_mesh(), ExcNotImplemented());
  const auto reference_cell = this->triangulation->get_reference_cells()[0];
  internal::DataOutRotationImplementation::ParallelData<dim, spacedim>
    thread_data(
      n_datasets,
      n_subdivisions,
      n_patches_per_circle,
      n_postprocessor_outputs,
      reference_cell.template get_default_linear_mapping<dim, spacedim>(),
      this->get_fes(),
      update_flags);
  std::vector<DataOutBase::Patch<patch_dim, patch_spacedim>> new_patches(
    n_patches_per_circle);
  for (unsigned int i = 0; i < new_patches.size(); ++i)
    {
      new_patches[i].n_subdivisions = n_subdivisions;
      new_patches[i].reference_cell = ReferenceCells::get_hypercube<dim + 1>();

      new_patches[i].data.reinit(
        n_datasets, Utilities::fixed_power<patch_dim>(n_subdivisions + 1));
    }

  // now build the patches in parallel
  WorkStream::run(
    all_cells.data(),
    all_cells.data() + all_cells.size(),
    [this](
      const cell_iterator *cell,
      internal::DataOutRotationImplementation::ParallelData<dim, spacedim>
                                                                 &data,
      std::vector<DataOutBase::Patch<patch_dim, patch_spacedim>> &my_patches) {
      this->build_one_patch(cell, data, my_patches);
    },
    [this](const std::vector<DataOutBase::Patch<patch_dim, patch_spacedim>>
             &new_patches) {
      internal::DataOutRotationImplementation::append_patch_to_list<dim,
                                                                    spacedim>(
        new_patches, this->patches);
    },
    thread_data,
    new_patches);
}



template <int dim, int spacedim>
typename DataOutRotation<dim, spacedim>::cell_iterator
DataOutRotation<dim, spacedim>::first_cell()
{
  return this->triangulation->begin_active();
}


template <int dim, int spacedim>
typename DataOutRotation<dim, spacedim>::cell_iterator
DataOutRotation<dim, spacedim>::next_cell(const cell_iterator &cell)
{
  // convert the iterator to an
  // active_iterator and advance
  // this to the next active cell
  typename Triangulation<dim, spacedim>::active_cell_iterator active_cell =
    cell;
  ++active_cell;
  return active_cell;
}



// explicit instantiations
#include "numerics/data_out_rotation.inst"


DEAL_II_NAMESPACE_CLOSE
