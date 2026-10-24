#include <unordered_set>

#include <assert.hpp>

#include "Xped/Util/Macros.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"
#include "Xped/Symmetry/S1xS2.hpp"
#include "Xped/Symmetry/CombSym.hpp"

#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/BlockUnaryOp.hpp"
#include "Xped/Core/CoeffBinaryOp.hpp"
#include "Xped/Core/CoeffUnaryOp.hpp"
#include "Xped/Core/DiagCoeffBinaryOp.hpp"
#include "Xped/Core/DiagCoeffUnaryOp.hpp"
#include "Xped/Core/Tensor.hpp"

#include "Xped/Core/TensorBase.hpp"

namespace Xped {

template <typename Derived>
template <bool>
typename TensorTraits<Derived>::Scalar TensorBase<Derived>::trace() XPED_CONST
{
    using RealScalar = typename ScalarTraits<Scalar>::Real;
    DEBUG_ASSERT(derived().coupledDomain() == derived().coupledCodomain());
    Scalar out = 0.;
    for(size_t i = 0; i < derived().sector().size(); i++) {
        out += PlainInterface::trace(derived().block(i)) * static_cast<RealScalar>(Symmetry::degeneracy(derived().sector(i)));
    }
    return out;
}

template <typename Derived>
typename TensorTraits<Derived>::Scalar TensorBase<Derived>::unweighted_trace() XPED_CONST
{
    DEBUG_ASSERT(derived().coupledDomain() == derived().coupledCodomain());
    Scalar out = 0.;
    for(size_t i = 0; i < derived().sector().size(); i++) { out += PlainInterface::trace(derived().block(i)); }
    return out;
}

template <typename Derived>
typename ScalarTraits<typename TensorTraits<Derived>::Scalar>::Real TensorBase<Derived>::maxNorm() XPED_CONST
{
    typename ScalarTraits<Scalar>::Real out = 0.;
    for(size_t i = 0; i < derived().sector().size(); i++) {
        if(out < PlainInterface::maxNorm(derived().block(i))) { out = PlainInterface::maxNorm(derived().block(i)); }
    }
    return out;
}

template <typename Derived>
typename ScalarTraits<typename TensorTraits<Derived>::Scalar>::Real TensorBase<Derived>::squaredNorm() XPED_CONST
{
    auto res = (*this * this->adjoint()).trace();
    DEBUG_ASSERT(std::abs(std::imag(res)) < 1.e-12);
    return std::real(res);
}

template <typename Derived>
typename ScalarTraits<typename TensorTraits<Derived>::Scalar>::Real
TensorBase<Derived>::maxCoeff(std::size_t& max_block, PlainInterface::MIndextype& max_row, PlainInterface::MIndextype& max_col) XPED_CONST
{
    typename ScalarTraits<Scalar>::Real out = 0.;
    for(size_t i = 0; i < derived().sector().size(); i++) {
        PlainInterface::MIndextype max_row_tmp;
        PlainInterface::MIndextype max_col_tmp;
        if(out < PlainInterface::maxCoeff(derived().block(i), max_row_tmp, max_col_tmp)) {
            out = PlainInterface::maxCoeff(derived().block(i), max_row, max_col);
            max_block = i;
        }
    }
    return out;
}

template <typename Derived>
XPED_CONST AdjointOp<Derived> TensorBase<Derived>::adjoint() XPED_CONST
{
    return AdjointOp<Derived>(derived());
}

template <typename Derived>
template <typename ReturnScalar>
XPED_CONST CoeffUnaryOp<Derived, ReturnScalar> TensorBase<Derived>::unaryExpr(const std::function<ReturnScalar(Scalar)>& coeff_func) XPED_CONST
{
    return CoeffUnaryOp<Derived, ReturnScalar>(derived(), coeff_func);
}

template <typename Derived>
XPED_CONST BlockUnaryOp<Derived> TensorBase<Derived>::unaryExpr(const std::function<MatrixType(const MatrixType&)>& coeff_func) XPED_CONST
{
    return BlockUnaryOp<Derived>(derived(), coeff_func);
}

template <typename Derived>
Derived& TensorBase<Derived>::operator+=(const Scalar offset)
{
    derived() = unaryExpr([offset](const Scalar s) { return offset + s; });
    return derived();
}

template <typename Derived>
Derived& TensorBase<Derived>::operator-=(const Scalar offset)
{
    derived() = unaryExpr([offset](const Scalar s) { return s - offset; });
    return derived();
}

template <typename Derived>
Derived& TensorBase<Derived>::operator*=(const Scalar scale)
{
    derived() = unaryExpr([scale](const Scalar s) { return scale * s; });
    return derived();
}

template <typename Derived>
Derived& TensorBase<Derived>::operator/=(const Scalar divisor)
{
    derived() = unaryExpr([divisor](const Scalar s) { return s / divisor; });
    return derived();
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived, typename TensorTraits<Derived>::Scalar> TensorBase<Derived>::sqrt() XPED_CONST
{
    return unaryExpr<typename TensorTraits<Derived>::Scalar>([](const Scalar s) { return std::sqrt(s); });
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived, typename TensorTraits<Derived>::Scalar> TensorBase<Derived>::inv() XPED_CONST
{
    return unaryExpr<typename TensorTraits<Derived>::Scalar>([](const Scalar s) { return 1. / s; });
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived, typename TensorTraits<Derived>::Scalar> TensorBase<Derived>::square() XPED_CONST
{
    return unaryExpr<typename TensorTraits<Derived>::Scalar>([](const Scalar s) { return std::pow(s, 2); });
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived, typename ScalarTraits<typename TensorTraits<Derived>::Scalar>::Real> TensorBase<Derived>::abs() XPED_CONST
{
    return unaryExpr<typename ScalarTraits<typename TensorTraits<Derived>::Scalar>::Real>([](const Scalar s) { return std::abs(s); });
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived, typename ScalarTraits<typename TensorTraits<Derived>::Scalar>::Real> TensorBase<Derived>::real() XPED_CONST
{
    return unaryExpr<typename ScalarTraits<typename TensorTraits<Derived>::Scalar>::Real>([](const Scalar s) { return std::real(s); });
}

template <typename Derived>
XPED_CONST CoeffUnaryOp<Derived, typename ScalarTraits<typename TensorTraits<Derived>::Scalar>::Real> TensorBase<Derived>::imag() XPED_CONST
{
    return unaryExpr<typename ScalarTraits<typename TensorTraits<Derived>::Scalar>::Real>([](const Scalar s) { return std::imag(s); });
}

    
template <typename Derived>
template <typename OtherScalar>
XPED_CONST CoeffUnaryOp<Derived, OtherScalar> TensorBase<Derived>::cast() XPED_CONST
{
    return unaryExpr<OtherScalar>([](const Scalar s) { return static_cast<OtherScalar>(s); });
}

template <typename Derived>
XPED_CONST BlockUnaryOp<Derived> TensorBase<Derived>::msqrt() XPED_CONST
{
    return unaryExpr([](const MatrixType& m) { return PlainInterface::msqrt(m); });
}

template <typename Derived>
XPED_CONST BlockUnaryOp<Derived> TensorBase<Derived>::mexp(Scalar factor) XPED_CONST
{
    return unaryExpr([factor](const MatrixType& m) { return PlainInterface::mexp(m, factor); });
}

template <typename Derived>
XPED_CONST DiagCoeffUnaryOp<Derived> TensorBase<Derived>::diagUnaryExpr(const std::function<Scalar(Scalar)>& coeff_func) XPED_CONST
{
    return DiagCoeffUnaryOp<Derived>(derived(), coeff_func);
}

template <typename Derived>
XPED_CONST DiagCoeffUnaryOp<Derived> TensorBase<Derived>::diag_inv() XPED_CONST
{
    return diagUnaryExpr([](const Scalar s) { return 1. / s; });
}

template <typename Derived>
XPED_CONST DiagCoeffUnaryOp<Derived> TensorBase<Derived>::diag_sqrt() XPED_CONST
{
    return diagUnaryExpr([](const Scalar s) { return std::sqrt(s); });
}

template <typename Derived>
template <typename OtherDerived>
XPED_CONST DiagCoeffBinaryOp<Derived, OtherDerived>
TensorBase<Derived>::diagBinaryExpr(XPED_CONST TensorBase<OtherDerived>& other, const std::function<Scalar(Scalar, Scalar)>& coeff_func) XPED_CONST
{
    return DiagCoeffBinaryOp<Derived, OtherDerived>(derived(), other.derived(), coeff_func);
}

template <typename Derived>
template <typename OtherDerived>
XPED_CONST CoeffBinaryOp<Derived, OtherDerived> TensorBase<Derived>::binaryExpr(
    XPED_CONST TensorBase<OtherDerived>& other,
    const std::function<std::common_type_t<Scalar, typename TensorTraits<OtherDerived>::Scalar>(Scalar, typename TensorTraits<OtherDerived>::Scalar)>&
        coeff_func) XPED_CONST
{
    return CoeffBinaryOp<Derived, OtherDerived>(derived(), other.derived(), coeff_func);
}

template <typename Derived>
template <typename OtherDerived>
Derived& TensorBase<Derived>::operator+=(XPED_CONST TensorBase<OtherDerived>& other)
{
    derived() = binaryExpr(other, [](const Scalar s1, const Scalar s2) { return s1 + s2; });
    return derived();
}

template <typename Derived>
template <typename OtherDerived>
Derived& TensorBase<Derived>::operator-=(XPED_CONST TensorBase<OtherDerived>& other)
{
    derived() = binaryExpr(other, [](const Scalar s1, const Scalar s2) { return s1 - s2; });
    return derived();
}

template <typename Derived>
template <bool, typename OtherDerived>
Tensor<std::common_type_t<typename TensorTraits<Derived>::Scalar, typename TensorTraits<OtherDerived>::Scalar>,
       TensorTraits<Derived>::Rank,
       TensorTraits<typename std::remove_const<std::remove_reference_t<OtherDerived>>::type>::CoRank,
       typename TensorTraits<Derived>::Symmetry,
       false,
       typename TensorTraits<Derived>::AllocationPolicy>
TensorBase<Derived>::operator*(XPED_CONST TensorBase<OtherDerived>& other) XPED_CONST
{
    using ResScalar = std::common_type_t<typename TensorTraits<Derived>::Scalar, typename TensorTraits<OtherDerived>::Scalar>;
    typedef typename std::remove_const<std::remove_reference_t<OtherDerived>>::type OtherDerived_;
    static_assert(CoRank == TensorTraits<OtherDerived_>::Rank);
    auto& derived_ref = derived();
    auto& other_derived_ref = other.derived();
    // fmt::print("world={}, other.world={}\n", derived_ref.world()->comm, other_derived_ref.world()->comm);
    DEBUG_ASSERT(derived_ref.world() == other_derived_ref.world());
    DEBUG_ASSERT(derived_ref.coupledCodomain() == other_derived_ref.coupledDomain());

    Tensor<ResScalar, Rank, TensorTraits<OtherDerived_>::CoRank, Symmetry, false, AllocationPolicy> Tout(
        derived_ref.uncoupledDomain(), other_derived_ref.uncoupledCodomain(), derived_ref.world());
    Tout.setZero();
    // std::unordered_set<typename Symmetry::qType> uniqueController;
    auto& other_dict = other_derived_ref.dict();
    // auto& this_dict = derived_ref.dict();
    for(size_t i = 0; i < derived_ref.sector().size(); i++) {
        // uniqueController.insert(derived_ref.sector(i));
        auto it = other_dict.find(derived_ref.sector(i));
        if(it == other_dict.end()) { continue; }
        auto it_out = Tout.dict().find(derived_ref.sector(i));
        // if(it_out == Tout.dict().end()) {
        // Tout.push_back(derived_ref.sector(i), PlainInterface::prod(derived_ref.block(i), other_derived_ref.block(it->second)));
        // SPDLOG_CRITICAL("({},{})x({},{})",
        //                 derived_ref.block(i).rows(),
        //                 derived_ref.block(i).cols(),
        //                 other_derived_ref.block(it->second).rows(),
        //                 other_derived_ref.block(it->second).cols());
        // } else {
#ifdef XPED_USE_EIGEN_MATRIX_LIB
        // Every sector appears at most once, and Tout owns fresh storage. Write
        // the product directly; output sectors absent from either operand retain
        // the zeros initialized above.
        auto&& out_block = Tout.block(it_out->second);
        PlainInterface::prod_into(out_block, derived_ref.block(i), other_derived_ref.block(it->second));
#else
        Tout.block(it_out->second) += PlainInterface::prod(derived_ref.block(i), other_derived_ref.block(it->second));
#endif
        // }
        // Tout.push_back(derived_ref.sector(i), PlainInterface::prod<Scalar>(derived_ref.block(i), other_derived_ref.block(it->second)));
        // Tout.block_[i] = T1.block_[i] * T2.block_[it->second];
    }
    // for(size_t i = 0; i < other_derived_ref.sector().size(); i++) {
    //     if(auto it = uniqueController.find(other_derived_ref.sector(i)); it != uniqueController.end()) { continue; }
    //     auto it = this_dict.find(other_derived_ref.sector(i));
    //     if(it == this_dict.end()) { continue; }
    //     // Tout.push_back(other_derived_ref.sector(i), Plain::template prod<Scalar>(derived_ref.block(it->second), other_derived_ref.block(i)));
    //     Tout.push_back(other_derived_ref.sector(i), PlainInterface::prod<Scalar>(derived_ref.block(it->second), other_derived_ref.block(i)));
    // }
    return Tout;
}

} // namespace Xped

#if __has_include("TensorBase.gen.cpp")
#    include "TensorBase.gen.cpp"
#endif
