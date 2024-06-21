// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2023 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_lazy_h
#define dealii_lazy_h


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/mutex.h>

#include <atomic>
#include <mutex>
#include <optional>
#include <type_traits>


DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup threads
 * @{
 */

/**
 * This class is a wrapper that provides a convenient mechanism for lazy
 * initialization of the contained object on first use. The class ensures
 * that on-demand initialization of some expensive data structure happens
 * (a) exactly once in a thread-safe manner, and that (b) subsequent checks
 * in hot paths are cheap.
 *
 * Lazy<T> is closely modeled after the `std::optional` interface providing
 * a `reset()` and `value()` method, but also and extending it with two
 * methods: `ensure_initialized(creator)` which, as the name suggests,
 * ensures that the contained object is properly initialized. If the
 * `Lazy<T>` happens happens to contain no value yet, it initializes the
 * wrapped object by calling the `creator()` function object and storing
 * the return value. In addition a `value_or_initialize(creator)` function
 * is provided that, similarly, ensures that the object is properly
 * initialized and then returns a reference to the contained value.
 *
 * Example usage could look like the following, where the `FE` class stores a
 * matrix that is expensive to compute and so we do not want to do it unless it
 * is actually needed. As a consequence, rather than storing a matrix, we store
 * a `Lazy<FullMatrix<double>>` that by default is empty; whenever the matrix is
 * first requested, we create it and store it for later reuse:
 * ```
 * template<...>
 * class FE
 * {
 * public:
 *   const FullMatrix<double> & get_prolongation_matrix() const
 *   {
 *     prolongation_matrix.ensure_initialized([&](){
 *       // Some expensive operation initializing the prolongation matrix
 *       // that we only want to perform once and when necessary.
 *       });
 *     return prolongation_matrix.value();
 *   }
 *
 * private:
 *   Lazy<FullMatrix<double>> prolongation_matrix;
 * };
 * ```
 *
 * @note Conceptually, this class is not so different from
 *   [std::future](https://en.cppreference.com/w/cpp/thread/future), which
 *   can also be used to represent a possibly-not-yet-available value on which
 *   one can wait when used with the "deferred" policy of
 *   [std::async](https://en.cppreference.com/w/cpp/thread/async).
 *   In particular, the following code could be used in place
 *   of the one above:
 * ```
 * template<...>
 * class FE
 * {
 * public:
 *   FE () {
 *     prolongation_matrix = std::async(std::launch::deferred,
 *       [&](){
 *       // Some expensive operation initializing the prolongation matrix
 *       // that we only want to perform once and when necessary.
 *       });
 *   }
 *
 *   const FullMatrix<double> & get_prolongation_matrix() const
 *   {
 *     return prolongation_matrix.get();
 *   }
 *
 * private:
 *   std::future<FullMatrix<double>> prolongation_matrix;
 * };
 * ```
 *   The difference to what Lazy does is that for Lazy, the action must be
 *   specified in the place where we want to access the deferred computation's
 *   result. In contrast, in the scheme with `std::future` and `std::async`,
 *   the action has to be provided at the point where the `std::future`
 *   object is initialized. Both are valid approaches and, depending on
 *   context, can usefully be employed. The difference is simply in what
 *   kind of information the provided lambda function can capture: Is it
 *   the environment available at the time the constructor is run, or the
 *   environment available at the time the access function is run. The latter
 *   has the advantage that the information captured is always up to date,
 *   whereas in the scheme with `std::async`, one has to be careful not to
 *   capture information in the lambda function that could be changed by later
 *   calls to member functions but before the lambda function is finally
 *   evaluated in the getter function.
 *
 * @dealiiConceptRequires{std::is_move_constructible_v<T> &&
                          std::is_move_assignable_v<T >}
 */
template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
class Lazy
{
public:
  /**
   * Default Constructor.
   */
  Lazy();


  /**
   * Copy constructor. If the `other` object contains an initialized
   * value, then that value will be copied into the current object. If the
   * `other` object is uninitialized, then the current object will be as well.
   */
  Lazy(const Lazy &other);


  /**
   * Move constructor. If the `other` object contains an initialized
   * value, then that value will be moved into the current object, and the
   * `other` object will end up being empty (as if default initialized). If the
   * `other` object is uninitialized, then the current object will be as well.
   */
  Lazy(Lazy &&other) noexcept;


  /**
   * Copy assignment. If the `other` object contains an initialized
   * value, then that value will be copied into the current object. If the
   * `other` object is uninitialized, then the current object will be as well.
   *
   * Any content of the current object is lost in the assignment.
   */
  Lazy &
  operator=(const Lazy &other);


  /**
   * Move assignment. If the `other` object contains an initialized
   * value, then that value will be moved into the current object, and the
   * `other` object will end up being empty (as if default initialized). If the
   * `other` object is uninitialized, then the current object will be as well.
   *
   * Any content of the current object is lost in the move assignment.
   */
  Lazy &
  operator=(Lazy &&other) noexcept;


  /**
   * Reset the Lazy<T> object to an uninitialized state.
   */
  void
  reset() noexcept;


  /**
   * Initialize the wrapped object.
   *
   * If the contained object is already initialized this function simply
   * returns and does nothing.
   *
   * If, instead, the object has not yet been initialized then the @p
   * creator function object (oftentimes a lambda function) is called to
   * initialize the contained object.
   *
   * This operation is thread safe: The ensure_initialized() method
   * guarantees that the creator function object is only called once on one
   * of the calling threads and that after completion the initialization
   * result (which is stored in the std::optional) is visible on all
   * threads.
   *
   * @dealiiConceptRequires{std::is_invocable_r_v<T, Callable>}
   */
  template <typename Callable>
  void
  ensure_initialized(const Callable &creator) const
    DEAL_II_CXX20_REQUIRES((std::is_invocable_r_v<T, Callable>));


  /**
   * Returns true if the contained object has been initialized, otherwise
   * false.
   */
  bool
  has_value() const;


  /**
   * Return a const reference to the contained object.
   *
   * @pre The object has been initialized with a call to
   * ensure_initialized() or value_or_initialized().
   */
  const T &
  value() const;


  /**
   * Return a reference to the contained object.
   *
   * @pre The object has been initialized with a call to
   * ensure_initialized() or value_or_initialize().
   */
  T &
  value();


  /**
   * If the underlying object is initialized the function simply returns a
   * const reference to the contained value. Otherwise, the @p creator()
   * function object is called to initialize the object first.
   *
   * This function mimics the syntax of the std::optional<T> interface and
   * is functionally equivalent to calling ensure_initialized() followed by
   * value().
   *
   * @note This method can be called from a context where the Lazy<T>
   * wrapper itself is marked const. FIXME
   *
   * @post The underlying object is initialized, meaning, has_value()
   * returns true.
   *
   * @dealiiConceptRequires{std::is_invocable_r_v<T, Callable>}
   */
  template <typename Callable>
  const T &
  value_or_initialize(const Callable &creator) const
    DEAL_II_CXX20_REQUIRES((std::is_invocable_r_v<T, Callable>));


  /**
   * Variant of above function that returns a non-const reference.
   *
   * @dealiiConceptRequires{std::is_invocable_r_v<T, Callable>}
   */
  template <typename Callable>
  DEAL_II_ALWAYS_INLINE inline T &
  value_or_initialize(const Callable &creator)
    DEAL_II_CXX20_REQUIRES((std::is_invocable_r_v<T, Callable>));


  /**
   * Compute the memory consumption of this structure.
   */
  std::size_t
  memory_consumption() const;

private:
  /**
   * The lazily initialized object stored as a std::optional<T>.
   */
  mutable std::optional<T> object;


  /**
   * An atomic bool used for checking whether the object is initialized in
   * a thread-safe manner.
   */
  mutable std::atomic<bool> object_is_initialized;


  /**
   * A mutex used for protecting the initialization of the object.
   */
  mutable Threads::Mutex initialization_mutex;
};

/**
 * @}
 */


// ------------------------------- inline functions --------------------------


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
inline Lazy<T>::Lazy()
  : object_is_initialized(false)
{}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
inline Lazy<T>::Lazy(const Lazy &other)
  : object(other.object)
{
  object_is_initialized.store(other.object_is_initialized.load());
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
inline Lazy<T>::Lazy(Lazy &&other) noexcept
  : object(std::move(other.object))
{
  object_is_initialized.store(other.object_is_initialized.load());

  // Mark the other object as uninitialized. This is marginally non-trivial
  // because moving from std::optional<T> does *not* result in an empty
  // std::optional<T> but instead one that does contain a T, but one that
  // has been moved from -- typically something akin to a default-initialized
  // T. That seems undesirable, so reset everything to an empty state.
  other.object_is_initialized.store(false);
  other.object.reset();
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
inline Lazy<T> &Lazy<T>::operator=(const Lazy &other)
{
  object = other.object;
  object_is_initialized.store(other.object_is_initialized.load());
  return *this;
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
inline Lazy<T> &Lazy<T>::operator=(Lazy &&other) noexcept
{
  object = std::move(other.object);
  object_is_initialized.store(other.object_is_initialized.load());

  // Mark the other object as uninitialized. This is marginally non-trivial
  // because moving from std::optional<T> does *not* result in an empty
  // std::optional<T> but instead one that does contain a T, but one that
  // has been moved from -- typically something akin to a default-initialized
  // T. That seems undesirable, so reset everything to an empty state.
  other.object_is_initialized.store(false);
  other.object.reset();

  return *this;
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
inline void Lazy<T>::reset() noexcept
{
  object_is_initialized.store(false);
  object.reset();
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
template <typename Callable>
inline DEAL_II_ALWAYS_INLINE
  void Lazy<T>::ensure_initialized(const Callable &creator) const
  DEAL_II_CXX20_REQUIRES((std::is_invocable_r_v<T, Callable>))
{
  //
  // Use Schmidt's double checking [1] for checking and initializing the
  // object.
  //
  // [1] https://en.wikipedia.org/wiki/Double-checked_locking
  //

  //
  // Check the object_is_initialized atomic with "acquire" semantics [1].
  //
  // This ensures that (a) all subsequent reads (of the object) are
  // ordered after this check, and that (b) all writes to the object
  // before the atomic bool was set to true with "release" semantics are
  // visible on this thread.
  //
  // [1]
  // https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
  //
  if (!object_is_initialized.load(std::memory_order_acquire))
#ifdef DEAL_II_HAVE_CXX20
    [[unlikely]]
#endif
    {
      std::lock_guard<std::mutex> lock(initialization_mutex);

      //
      // Check again. If this thread won the race to the lock then we
      // initialize the object. Otherwise another thread has already
      // initialized the object and flipped the object_is_initialized
      // bit. (Here, the initialization_mutex ensures consistent ordering
      // with a memory fence, so we will observe the updated bool without
      // acquire semantics.)
      //
      if (!object_is_initialized.load(std::memory_order_relaxed))
        {
          Assert(object.has_value() == false, ExcInternalError());
          object.emplace(std::move(creator()));

          //
          // Flip the object_is_initialized boolean with "release"
          // semantics [1].
          //
          // This ensures that the above move is visible on all threads
          // before checking the atomic bool with acquire semantics.
          //
          object_is_initialized.store(true, std::memory_order_release);
        }
    }

  Assert(
    object.has_value(),
    dealii::ExcMessage(
      "The internal std::optional<T> object does not contain a valid object "
      "even though we have just initialized it."));
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
inline DEAL_II_ALWAYS_INLINE bool Lazy<T>::has_value() const
{
  //
  // In principle it would be sufficient to solely check the atomic<bool>
  // object_is_initialized because the load() is performed with "acquire"
  // semantics. But just in case let's check the object.has_value() boolean
  // as well:
  //
  return (object_is_initialized && object.has_value());
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
inline DEAL_II_ALWAYS_INLINE const T &Lazy<T>::value() const
{
  Assert(
    object_is_initialized && object.has_value(),
    dealii::ExcMessage(
      "value() has been called but the contained object has not been "
      "initialized. Did you forget to call 'ensure_initialized()' first?"));

  return object.value();
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
inline DEAL_II_ALWAYS_INLINE T &Lazy<T>::value()
{
  Assert(
    object_is_initialized && object.has_value(),
    dealii::ExcMessage(
      "value() has been called but the contained object has not been "
      "initialized. Did you forget to call 'ensure_initialized()' first?"));

  return object.value();
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
template <typename Callable>
inline DEAL_II_ALWAYS_INLINE const T &Lazy<T>::value_or_initialize(
  const Callable &creator) const
  DEAL_II_CXX20_REQUIRES((std::is_invocable_r_v<T, Callable>))
{
  ensure_initialized(creator);
  return object.value();
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
template <typename Callable>
inline DEAL_II_ALWAYS_INLINE T &Lazy<T>::value_or_initialize(
  const Callable &creator)
  DEAL_II_CXX20_REQUIRES((std::is_invocable_r_v<T, Callable>))
{
  ensure_initialized(creator);
  return object.value();
}


template <typename T>
DEAL_II_CXX20_REQUIRES((std::is_move_constructible_v<T> &&
                        std::is_move_assignable_v<T>))
std::size_t Lazy<T>::memory_consumption() const
{
  return MemoryConsumption::memory_consumption(object) + //
         sizeof(*this) - sizeof(object);
}


DEAL_II_NAMESPACE_CLOSE
#endif
