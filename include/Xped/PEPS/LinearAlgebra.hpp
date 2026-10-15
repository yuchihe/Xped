#ifndef XPED_PEPS_LINEAR_ALGEBRA_HPP_
#define XPED_PEPS_LINEAR_ALGEBRA_HPP_

#include <tuple>

#include "Xped/Util/Macros.hpp"

#include "Xped/Core/Tensor.hpp"

#include "Xped/AD/ADTensor.hpp"

#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/OneSiteObservable.hpp"
#include "Xped/PEPS/TMatrix.hpp"
#include "Xped/PEPS/TwoSiteObservable.hpp"

namespace Xped {

namespace detail {
// The dense sliced kernel uses real U0 components. Symmetric, complex,
// fused-edge and all-out environments retain the tensor contraction path.
template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, typename OpScalar>
inline constexpr bool supports_sliced_energy =
#ifdef XPED_USE_EIGEN_MATRIX_LIB
    std::is_same_v<Scalar, double> && std::is_same_v<OpScalar, double> && Symmetry::Nq == 0 && TRank == 2 && !ALL_OUT_LEGS;
#else
    false;
#endif
} // namespace detail

template <typename Scalar,
          typename Symmetry,
          std::size_t TRank,
          bool ALL_OUT_LEGS,
          bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts,
          typename OpScalar,
          bool HERMITIAN>
TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename OneSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>
avg(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env, OneSiteObservable<OpScalar, Symmetry, HERMITIAN>& op);

template <typename Scalar,
          typename Symmetry,
          std::size_t TRank,
          bool ALL_OUT_LEGS,
          bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts,
          typename OpScalar,
          bool HERMITIAN>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>, 4>
avg(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op);

// Exact interaction/physical/chi slicing. No precomputed RDM is needed by
// the supported kernel; other tensor types use the original contractions.
template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts, typename OpScalar, bool HERMITIAN>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>, 4>
avg_sliced(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env,
           TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op, std::size_t block_size = 8);

// Original RDM-based contraction. Disable the historical operator multiplet
// cutoff to obtain an untruncated reference for the exact sliced evaluator.
template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts, typename OpScalar, bool HERMITIAN>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>, 4>
avg_unsliced(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env,
             TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op, bool preserve_operator_multiplets = true);

template <typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename OneSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>
avg(XPED_CONST Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>& rho1, OneSiteObservable<OpScalar, Symmetry, HERMITIAN>& op);

template <typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>, 4>
avg(XPED_CONST Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>& rho, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op, Opts::Bond bond);

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "PEPS/LinearAlgebra.cpp"
#endif

#endif
