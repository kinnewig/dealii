// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 1998 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_quadrature_h
#define dealii_quadrature_h


#include <deal.II/base/config.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <array>
#include <memory>
#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup Quadrature
 * @{
 */

/**
 * Base class for quadrature formulae in arbitrary dimensions.
 * [Quadrature](https://en.wikipedia.org/wiki/Numerical_integration)
 * is a means to approximate an integral by evaluating the integrand
 * at specific points $\mathbf x_q$ and summing the point values with
 * specific weights $w_q$; that is, quadrature computes
 * @f{align*}{
 *   \int_K f(\mathbf x) \; dx
 *   \approx
 *   \sum_{q=0,\ldots,Q-1} f(\mathbf x_q) w_q.
 * @f}
 *
 * This class stores quadrature points $\mathbf x_q$ and weights $w_q$
 * for concrete "quadrature formulas" when $K$ (the domain we integrate
 * over) is a reference cell. That is, points and weights are expressed
 * in the coordinate system of
 * a reference cell (see the ReferenceCell class) and as such serves to
 * represent quadrature points and weights on the unit line segment
 * $[0,1]$ in 1d, on the unit square or unit triangle in 2d, as well as
 * the unit tetrahedron, cube, pyramid, and wedge reference cells in 3d.
 * Integration over concrete cells is done by coordinate transformation
 * to the reference cell represented by the current class.
 *
 * There are a number of derived classes, denoting concrete integration
 * formulae. Their names are prefixed by <tt>Q</tt>. Refer to the list of
 * derived classes for more details.
 *
 * At least for quadrilaterals and hexahedra (or, more precisely, since we work
 * on reference cells: for the unit square and the unit cube), quadrature
 * formulas are typically tensor products of one-dimensional formulas (see also
 * the section on implementation detail below).
 *
 * In order to allow for dimension independent programming, a quadrature
 * formula of dimension zero exists. Since an integral over zero dimensions is
 * the evaluation at a single point, any constructor of such a formula
 * initializes to a single quadrature point with weight one. Access to the
 * weight is possible, while access to the quadrature point is not permitted,
 * since a Point of dimension zero contains no information. The main purpose
 * of these formulae is their use in QProjector, which will create a useful
 * formula of dimension one out of them.
 *
 *
 * <h3>Mathematical background</h3>
 *
 * For each quadrature formula we denote by <tt>m</tt>, the maximal degree of
 * polynomials integrated exactly on the reference cell the quadrature
 * formula corresponds to. This number is given in the documentation
 * of each formula. The *order* of the integration error is <tt>m+1</tt>, that
 * is, the error is the size of the cell to the <tt>m+1</tt> by the
 * Bramble-Hilbert Lemma. The number <tt>m</tt> is to be found in the
 * documentation of each concrete formula. For the optimal formulae QGauss we
 * have $m = 2N-1$, where $N$ is the constructor parameter to QGauss. The tensor
 * product formulae are exact on tensor product polynomials of degree <tt>m</tt>
 * in each space direction, but they are still only of <tt>(m+1)</tt>st order.
 *
 *
 * <h3>Tensor product quadrature</h3>
 *
 * At least for hypercube reference cells (i.e., squares and cubes),
 * most integration formulae in more than one space dimension are tensor
 * products of quadrature formulae in one space dimension, or more generally
 * the tensor product of a formula in <tt>(dim-1)</tt> dimensions and one in
 * one dimension. There is a special constructor to generate a quadrature
 * formula from two others.  For example, the QGauss@<dim@> formulae include
 * <i>N<sup>dim</sup></i> quadrature points in <tt>dim</tt> dimensions, where
 * $N$ is the constructor parameter of QGauss.
 *
 *
 * <h3>Other uses of this class</h3>
 *
 * Quadrature objects are used in a number of places within deal.II where
 * integration is performed, most notably via the FEValues and related classes.
 * Some of these classes are also used in contexts where no integrals
 * are involved, but where functions need to be evaluated at specific
 * points, for example to evaluate the solution at individual points
 * or to create graphical output. Examples are the implementation of
 * VectorTools::point_value() and the DataOut and related classes (in
 * particular in connection with the DataPostprocessor class). In
 * such contexts, one often creates specific "Quadrature" objects in
 * which the "quadrature points" are simply the points (in the coordinate
 * system of the reference cell) at which one wants to evaluate the
 * solution. In these kinds of cases, the weights stored by the current
 * class are not used and the name "quadrature object" is interpreted
 * as "list of evaluation points".
 */
template <int dim>
class Quadrature : public EnableObserverPointer
{
public:
  /**
   * Define an alias for a quadrature that acts on an object of one dimension
   * less. For cells, this would then be a face quadrature. A sub quadrature of
   * a 0-dimensional quadrature is defined as still being 0-dimensional.
   */
  using SubQuadrature = Quadrature<dim == 0 ? 0 : dim - 1>;

  /**
   * Default constructor.
   */
  Quadrature();

  /**
   * Build this quadrature formula as the tensor product of a formula in a
   * dimension one less than the present and a formula in one dimension.
   * This constructor assumes (and tests) that constant functions are integrated
   * exactly, i.e. the sum of the quadrature weights is one.
   *
   * <tt>SubQuadrature<dim>::type</tt> expands to <tt>Quadrature<dim-1></tt>.
   */
  Quadrature(const SubQuadrature &, const Quadrature<1> &);

  /**
   * Build this quadrature formula as the <tt>dim</tt>-fold tensor product of
   * a formula in one dimension.
   *
   * Assuming that the points in the one-dimensional rule are in ascending
   * order, the points of the resulting rule are ordered lexicographically
   * with <i>x</i> running fastest.
   *
   * In order to avoid a conflict with the copy constructor in 1d, we let the
   * argument be a 0d quadrature formula for dim==1, and a 1d quadrature
   * formula for all other space dimensions.
   *
   * This constructor does not require that constant functions are integrated
   * exactly. Therefore, it is appropriate if the one-dimensional formula
   * is defined with respect to a weighting function.
   *
   * If dim == 0, the resulting quadrature formula will be a single Point<0>
   * having unit weight.
   */
  explicit Quadrature(const Quadrature<dim != 1 ? 1 : 0> &quadrature_1d);

  /**
   * Copy constructor.
   */
  Quadrature(const Quadrature<dim> &q);

  /**
   * Move constructor. Construct a new quadrature object by transferring the
   * internal data of another quadrature object.
   */
  Quadrature(Quadrature<dim> &&) noexcept = default;

  /**
   * Construct a quadrature formula from given vectors of quadrature points
   * (which should really be in the unit cell) and the corresponding weights.
   * You will want to have the weights sum up to one, but this is not checked.
   */
  Quadrature(const std::vector<Point<dim>> &points,
             const std::vector<double>     &weights);

  /**
   * Construct a quadrature formula from given vectors of quadrature points
   * (which should really be in the unit cell) and the corresponding weights,
   * moving the points and weights into the present object.
   */
  Quadrature(std::vector<Point<dim>> &&points, std::vector<double> &&weights);

  /**
   * Construct a dummy quadrature formula from a list of points, with weights
   * set to infinity. The resulting object is therefore not meant to actually
   * perform integrations, but rather to be used with FEValues objects in
   * order to find the position of some points (the quadrature points in this
   * object) on the transformed cell in real space.
   */
  Quadrature(const std::vector<Point<dim>> &points);

  /**
   * Constructor for a one-point quadrature. Sets the weight of this point to
   * one.
   */
  Quadrature(const Point<dim> &point);

  /**
   * Virtual destructor.
   */
  virtual ~Quadrature() override = default;

  /**
   * Assignment operator. Copies contents of #weights and #quadrature_points
   * as well as size.
   */
  Quadrature &
  operator=(const Quadrature<dim> &);

  /**
   * Move assignment operator. Moves all data from another quadrature object
   * to this object.
   */
  Quadrature &
  operator=(Quadrature<dim> &&) = default; // NOLINT

  /**
   * Test for equality of two quadratures.
   */
  bool
  operator==(const Quadrature<dim> &p) const;

  /**
   * Set the quadrature points and weights to the values provided in the
   * arguments. The weights array is allowed to be empty, in which case the
   * weights are set to infinity. The resulting object is therefore not meant
   * to actually perform integrations, but rather to be used with FEValues
   * objects in order to find the position of some points (the quadrature
   * points in this object) on the transformed cell in real space.
   */
  void
  initialize(const ArrayView<const Point<dim>> &points,
             const ArrayView<const double>     &weights = {});

  /**
   * Number of quadrature points.
   */
  unsigned int
  size() const;

  /**
   * Return if quadrature is empty.
   */
  bool
  empty() const;

  /**
   * Return the <tt>i</tt>th quadrature point.
   */
  const Point<dim> &
  point(const unsigned int i) const;

  /**
   * Return a reference to the whole array of quadrature points.
   */
  const std::vector<Point<dim>> &
  get_points() const;

  /**
   * Return the weight of the <tt>i</tt>th quadrature point.
   */
  double
  weight(const unsigned int i) const;

  /**
   * Return a reference to the whole array of weights.
   */
  const std::vector<double> &
  get_weights() const;

  /**
   * Determine an estimate for the memory consumption (in bytes) of this
   * object.
   */
  std::size_t
  memory_consumption() const;

  /**
   * Write or read the data of this object to or from a stream for the purpose
   * of serialization using the [BOOST serialization
   * library](https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/index.html).
   */
  template <class Archive>
  void
  serialize(Archive &ar, const unsigned int version);

  /**
   * This function returns true if the quadrature object is a tensor product
   * of one-dimensional formulas and the quadrature points are sorted
   * lexicographically.
   */
  bool
  is_tensor_product() const;

  /**
   * In case the quadrature formula is a tensor product, this function
   * returns the @p dim one-dimensional basis objects.
   * Otherwise, calling this function is not allowed.
   *
   * For @p dim equal to one, we can not return the std::array as a const
   * reference and have to return it by value. In this case, the array
   * will always contain a single element (@p this).
   *
   * @note The actual return type of this function is
   * @code
   * std::conditional_t<dim == 1,
   *                    std::array<Quadrature<1>, dim>,
   *                    const std::array<Quadrature<1>, dim> &>
   * @endcode
   * The type is abbreviated in the online documentation to improve
   * readability of this page.
   */
#ifndef DOXYGEN
  std::conditional_t<dim == 1,
                     std::array<Quadrature<1>, dim>,
                     const std::array<Quadrature<1>, dim> &>
#else
  const std::array<Quadrature<1>, dim> &
#endif
  get_tensor_basis() const;

protected:
  /**
   * Constructor.
   *
   * This constructor is marked as explicit to avoid involuntary accidents
   * like in <code>hp::QCollection@<dim@> q_collection(3)</code> where
   * <code>hp::QCollection@<dim@> q_collection(QGauss@<dim@>(3))</code> was
   * meant. Nonetheless, it is easy to accidentally write
   * @code
   *   Quadrature<dim> quadrature(3);
   * @endcode
   * where QGauss was meant. As a consequence, this constructor is `protected`
   * and so only available to derived classes initializing their base class.
   */
  explicit Quadrature(const unsigned int n_quadrature_points);

  /**
   * List of quadrature points. To be filled by the constructors of derived
   * classes.
   */
  std::vector<Point<dim>> quadrature_points;

  /**
   * List of weights of the quadrature points.  To be filled by the
   * constructors of derived classes.
   */
  std::vector<double> weights;

  /**
   * Indicates if this object represents quadrature formula that is a tensor
   * product of one-dimensional formulas.
   * This flag is set if dim==1 or the constructors taking a Quadrature<1>
   * (and possibly a Quadrature<dim-1> object) is called. This implies
   * that the quadrature points are sorted lexicographically.
   */
  bool is_tensor_product_flag;

  /**
   * Stores the one-dimensional tensor basis objects in case this object
   * can be represented by a tensor product.
   */
  std::unique_ptr<std::array<Quadrature<1>, dim>> tensor_basis;
};


/**
 * Quadrature formula implementing anisotropic distributions of quadrature
 * points on the reference cell. To this end, the tensor product of
 * <tt>dim</tt> one-dimensional quadrature formulas is generated.
 *
 * @note Each constructor can only be used in the dimension matching the
 * number of arguments.
 */
template <int dim>
class QAnisotropic : public Quadrature<dim>
{
public:
  /**
   * Constructor for a one-dimensional formula. This one just copies the given
   * quadrature rule.
   */
  QAnisotropic(const Quadrature<1> &qx);

  /**
   * Constructor for a two-dimensional formula.
   */
  QAnisotropic(const Quadrature<1> &qx, const Quadrature<1> &qy);

  /**
   * Constructor for a three-dimensional formula.
   */
  QAnisotropic(const Quadrature<1> &qx,
               const Quadrature<1> &qy,
               const Quadrature<1> &qz);
};


/**
 * Quadrature formula constructed by iteration of another quadrature formula
 * in each direction. In more than one space dimension, the resulting
 * quadrature formula is constructed in the usual way by building the tensor
 * product of the respective iterated quadrature formula in one space
 * dimension.
 *
 * In one space dimension, the given base formula is copied and scaled onto a
 * given number of subintervals of length <tt>1/n_copies</tt>. If the
 * quadrature formula uses both end points of the unit interval, then in the
 * interior of the iterated quadrature formula there would be quadrature
 * points which are used twice; we merge them into one with a weight which is
 * the sum of the weights of the left- and the rightmost quadrature point.
 *
 * Since all dimensions higher than one are built up by tensor products of one
 * dimensional and <tt>dim-1</tt> dimensional quadrature formulae, the
 * argument given to the constructor needs to be a quadrature formula in one
 * space dimension, rather than in <tt>dim</tt> dimensions.
 *
 * The aim of this class is to provide a low order formula, where the error
 * constant can be tuned by increasing the number of quadrature points. This
 * is useful in integrating non-differentiable functions on cells.
 */
template <int dim>
class QIterated : public Quadrature<dim>
{
public:
  /**
   * Constructor. Iterate the given quadrature formula <tt>n_copies</tt> times
   * in each direction. The result is a tensor product quadrature formula
   * defined on the unit hypercube (i.e., the line segment, unit square, or
   * unit cube in 1d, 2d, and 3d respectively).
   */
  QIterated(const Quadrature<1> &base_quadrature, const unsigned int n_copies);

  /**
   * Constructor. Iterate the given quadrature formula on the given subintervals
   * defined by adjacent points in @p intervals in each direction. The resulting
   * quadrature rule will have `base_quadrature.size() * (intervals.size() - 1)`
   * quadrature points if no quadrature point of `base_quadrature` is positioned
   * on the boundaries. The result is a tensor product quadrature formula
   * defined on the unit hypercube (i.e., the line segment, unit square, or
   * unit cube in 1d, 2d, and 3d respectively).
   *
   * @note We require that `intervals.front() == 0` and `interval.back() == 1`.
   */
  QIterated(const Quadrature<1>         &base_quadrature,
            const std::vector<Point<1>> &intervals);

  /**
   * Exception
   */
  DeclExceptionMsg(ExcInvalidQuadratureFormula,
                   "The quadrature formula you provided cannot be used "
                   "as the basis for iteration.");
};



/** @} */

#ifndef DOXYGEN

// -------------------  inline and template functions ----------------


template <int dim>
inline unsigned int
Quadrature<dim>::size() const
{
  return weights.size();
}



template <int dim>
inline bool
Quadrature<dim>::empty() const
{
  return weights.empty();
}



template <int dim>
inline const Point<dim> &
Quadrature<dim>::point(const unsigned int i) const
{
  AssertIndexRange(i, size());
  return quadrature_points[i];
}



template <int dim>
double
Quadrature<dim>::weight(const unsigned int i) const
{
  AssertIndexRange(i, size());
  return weights[i];
}



template <int dim>
inline const std::vector<Point<dim>> &
Quadrature<dim>::get_points() const
{
  return quadrature_points;
}



template <int dim>
inline const std::vector<double> &
Quadrature<dim>::get_weights() const
{
  return weights;
}



template <int dim>
inline bool
Quadrature<dim>::is_tensor_product() const
{
  return is_tensor_product_flag;
}



template <int dim>
template <class Archive>
inline void
Quadrature<dim>::serialize(Archive &ar, const unsigned int)
{
  // forward to serialization
  // function in the base class.
  ar &static_cast<EnableObserverPointer &>(*this);

  ar &quadrature_points &weights;
}



/* -------------- declaration of explicit specializations ------------- */

template <>
Quadrature<0>::Quadrature(const unsigned int);
template <>
Quadrature<0>::Quadrature(const Quadrature<0>::SubQuadrature &,
                          const Quadrature<1> &);
template <>
Quadrature<0>::Quadrature(const Quadrature<1> &);
template <>
Quadrature<0>::Quadrature(const Point<0> &);

template <>
Quadrature<1>::Quadrature(const Quadrature<0> &, const Quadrature<1> &);

template <>
Quadrature<1>::Quadrature(const Quadrature<0> &);

template <>
QIterated<1>::QIterated(const Quadrature<1> &base_quadrature,
                        const unsigned int   n_copies);

#endif // DOXYGEN
DEAL_II_NAMESPACE_CLOSE

#endif
