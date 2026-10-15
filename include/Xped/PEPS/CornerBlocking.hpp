#ifndef XPED_CORNER_BLOCKING_HPP_
#define XPED_CORNER_BLOCKING_HPP_

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "Xped/AD/ADTensor.hpp"

namespace Xped::detail {

template <typename T>
const auto& corner_value(const T& tensor)
{
    if constexpr(T::IS_AD) { return tensor.val(); }
    else { return tensor; }
}

// Slice multiplicity spaces, never magnetic components of an irrep. Keeping
// the fusion trees and their dual flags makes this work for non-Abelian as
// well as Abelian tensors without converting them to dense physical arrays.
template <std::size_t Leg, typename T>
const auto& corner_leg_basis(const T& tensor)
{
    using Plain = std::remove_cvref_t<decltype(tensor.detach())>;
    if constexpr(Leg < Plain::rank()) { return tensor.uncoupledDomain()[Leg]; }
    else { return tensor.uncoupledCodomain()[Leg - Plain::rank()]; }
}

template <std::size_t Leg, typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
auto corner_with_basis(const Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy>& tensor,
                       const Qbasis<Symmetry, 1, AllocationPolicy>& basis)
{
    auto domain = tensor.uncoupledDomain();
    auto codomain = tensor.uncoupledCodomain();
    if constexpr(Leg < Rank) { domain[Leg] = basis; }
    else { codomain[Leg - Rank] = basis; }
    Tensor<Scalar, Rank, CoRank, Symmetry, false, AllocationPolicy> result(domain, codomain, tensor.world());
    result.setZero();
    return result;
}

#ifdef XPED_USE_EIGEN_MATRIX_LIB
// Visit matching blocks of the original and sliced reduced tensors. No
// recoupling or permutation of the complete enlarged corner is required.
template <std::size_t Leg, typename Full, typename Slice, typename F>
void corner_slice_blocks(Full& full, Slice& slice, std::size_t begin, F&& operation)
{
    using T = std::remove_cv_t<Full>;
    constexpr auto rank = T::rank();
    for(const auto& sector : slice.sector()) {
        for(const auto& small_domain : slice.coupledDomain().tree(sector)) {
            for(const auto& small_codomain : slice.coupledCodomain().tree(sector)) {
                auto domain = small_domain;
                auto codomain = small_codomain;
                std::size_t stride = 1, width, full_width, groups;
                if constexpr(Leg < rank) {
                    width = domain.dims[Leg];
                    full_width = corner_leg_basis<Leg>(full).inner_dim(domain.q_uncoupled[Leg]);
                    for(std::size_t i = 0; i < Leg; ++i) { stride *= domain.dims[i]; }
                    groups = domain.dim / (stride * width);
                    domain.dims[Leg] = full_width;
                    domain.computeDim();
                } else {
                    constexpr auto leg = Leg - rank;
                    width = codomain.dims[leg];
                    full_width = corner_leg_basis<Leg>(full).inner_dim(codomain.q_uncoupled[leg]);
                    for(std::size_t i = 0; i < leg; ++i) { stride *= codomain.dims[i]; }
                    groups = codomain.dim / (stride * width);
                    codomain.dims[leg] = full_width;
                    codomain.computeDim();
                }
                auto big = full.subMatrix(domain, codomain);
                auto small = slice.subMatrix(small_domain, small_codomain);
                for(std::size_t group = 0; group < groups; ++group) {
                    const auto offset = (group * full_width + begin) * stride;
                    const auto small_offset = group * width * stride;
                    if constexpr(Leg < rank) {
                        operation(big.middleRows(offset, width * stride), small.middleRows(small_offset, width * stride));
                    } else {
                        operation(big.middleCols(offset, width * stride), small.middleCols(small_offset, width * stride));
                    }
                }
            }
        }
    }
}

template <std::size_t Leg, typename T>
auto corner_slice(const T& tensor, const typename T::qType& q, std::size_t begin, std::size_t width)
{
    const auto& full_basis = corner_leg_basis<Leg>(tensor);
    if(width == 0 || begin + width > full_basis.inner_dim(q)) { throw std::invalid_argument("Invalid corner slice"); }
    std::remove_cvref_t<decltype(full_basis)> basis;
    if(full_basis.IS_CONJ()) { basis.SET_CONJ(); }
    auto trees = full_basis.tree(q);
    for(auto& tree : trees) { tree.dims[0] = width; tree.computeDim(); }
    basis.push_back(q, width, trees);
    auto result = corner_with_basis<Leg>(tensor, basis);
    corner_slice_blocks<Leg>(tensor, result, begin, [](auto&& whole, auto&& part) { part = whole; });
    return result;
}

template <std::size_t Leg, typename Full, typename Slice>
void corner_scatter_add(Full& full, const Slice& slice, std::size_t begin)
{
    corner_slice_blocks<Leg>(full, slice, begin, [](auto&& whole, auto&& part) { whole += part; });
}

// F<false> evaluates without a tape; F<true> replays one slice on a nested
// tape. F must preserve the sliced free leg, at OutLeg in its result, and
// capture shared AD inputs by value. Only input/output handles survive on
// the outer tape; all per-slice intermediates are released after replay.
template <std::size_t InLeg, std::size_t OutLeg, bool TRACK, typename Input, typename F>
auto blocked_corner(const Input& input, F function, std::size_t block_size = 8)
{
    if(block_size == 0) { throw std::invalid_argument("Corner block size must be positive"); }
    using PlainInput = std::decay_t<decltype(input.detach())>;
    using Output = std::decay_t<decltype(function.template operator()<false>(std::declval<const PlainInput&>()))>;
    using Symmetry = typename Input::Symmetry;
    const auto& basis = corner_leg_basis<InLeg>(input);
    Output value;
    bool initialized = false, conjugate = false;
    for(const auto& q : basis.qs()) {
        for(std::size_t begin = 0; begin < basis.inner_dim(q); begin += block_size) {
            const auto width = std::min(block_size, basis.inner_dim(q) - begin);
            auto slice = corner_slice<InLeg>(corner_value(input), q, begin, width);
            auto part = function.template operator()<false>(slice);
            if(!initialized) {
                conjugate = corner_leg_basis<OutLeg>(part).IS_CONJ() != basis.IS_CONJ();
                value = corner_with_basis<OutLeg>(part, conjugate ? basis.conj() : basis);
                initialized = true;
            }
            corner_scatter_add<OutLeg>(value, part, begin);
        }
    }
    if(!initialized) { throw std::invalid_argument("Cannot block an empty corner boundary"); }
    if constexpr(TRACK) {
        static_assert(Input::IS_AD);
        using ADInput = Tensor<typename Input::Scalar, PlainInput::rank(), PlainInput::corank(), Symmetry, true, typename Input::AllocationPolicy>;
        using ADOutput = Tensor<typename Output::Scalar, Output::rank(), Output::corank(), Symmetry, true, typename Output::AllocationPolicy>;
        ADOutput result(value);
        Xped::reverse_pass_callback_alloc([input, result, function, block_size, conjugate]() mutable {
            const auto& full_basis = corner_leg_basis<InLeg>(input);
            for(const auto& q : full_basis.qs()) {
                for(std::size_t begin = 0; begin < full_basis.inner_dim(q); begin += block_size) {
                    const auto width = std::min(block_size, full_basis.inner_dim(q) - begin);
                    stan::math::nested_rev_autodiff nested;
                    ADInput slice(corner_slice<InLeg>(corner_value(input), q, begin, width));
                    auto part = function.template operator()<true>(slice);
                    part.adj() = corner_slice<OutLeg>(result.adj(), conjugate ? Symmetry::conj(q) : q, begin, width);
                    stan::math::grad();
                    corner_scatter_add<InLeg>(input.adj(), slice.adj(), begin);
                }
            }
        });
        return result;
    } else {
        return value;
    }
}
#endif

} // namespace Xped::detail

#endif
