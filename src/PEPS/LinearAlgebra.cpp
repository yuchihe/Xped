#include "Xped/PEPS/LinearAlgebra.hpp"
#include "Xped/PEPS/EnergySlicing.hpp"

#include <limits>

#include "spdlog/spdlog.h"

namespace Xped {

template <typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename OneSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>
avg(XPED_CONST Tensor<Scalar, 1, 1, Symmetry, ENABLE_AD>& rho1, OneSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    using ObsScalar = typename OneSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar;
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o(op.data.pat);
    o.setConstant(0.);
    if(not op.MEASURE) { return o; }

    for(int x = 0; x < op.data.pat.Lx; ++x) {
        for(int y = 0; y < op.data.pat.Ly; ++y) {
            if(not op.data.pat.isUnique(x, y)) { continue; }
            if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                o(x, y) = rho1.twist(0).template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(op.data(x, y)).trace();
            } else {
                o(x, y) = std::real(rho1.twist(0).template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(op.data(x, y)).trace());
            }
            if constexpr(ENABLE_AD) {
                op.obs(x, y) = o(x, y).val();
            } else {
                op.obs(x, y) = o(x, y);
            }
        }
    }
    return o;
}

template <typename Scalar, typename Symmetry, bool ENABLE_AD, typename OpScalar, bool HERMITIAN>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>, 4>
avg(XPED_CONST Tensor<Scalar, 2, 2, Symmetry, ENABLE_AD>& rho, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op, Opts::Bond bond)
{
    using ObsScalar = typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar;

    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o_h(op.data_h.pat);
    o_h.setConstant(0.);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o_v(op.data_v.pat);
    o_v.setConstant(0.);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o_d1(op.data_d1.pat);
    o_d1.setConstant(0.);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o_d2(op.data_d2.pat);
    o_d2.setConstant(0.);
    if(not op.MEASURE) { return std::array{o_h, o_v, o_d1, o_d2}; }

    for(int x = 0; x < op.data_h.pat.Lx; ++x) {
        for(int y = 0; y < op.data_h.pat.Ly; ++y) {
            if(not op.data_h.pat.isUnique(x, y)) { continue; }
            if((bond & Opts::Bond::H) == Opts::Bond::H) {
                if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                    o_h(x, y) = rho.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_h(x, y)).trace();
                } else {
                    o_h(x, y) = std::real(
                        rho.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_h(x, y)).trace());
                }
                if constexpr(ENABLE_AD) {
                    op.obs_h(x, y) = o_h(x, y).val();
                } else {
                    op.obs_h(x, y) = o_h(x, y);
                }
            }
            if((bond & Opts::Bond::V) == Opts::Bond::V) {
                if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                    o_v(x, y) = rho.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_v(x, y)).trace();
                } else {
                    o_v(x, y) = std::real(
                        rho.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_v(x, y)).trace());
                }
                if constexpr(ENABLE_AD) {
                    op.obs_v(x, y) = o_v(x, y).val();
                } else {
                    op.obs_v(x, y) = o_v(x, y);
                }
            }
            if((bond & Opts::Bond::D1) == Opts::Bond::D1) {
                if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                    o_d1(x, y) = rho.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_d1(x, y)).trace();
                } else {
                    o_d1(x, y) = std::real(
                        rho.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_d1(x, y)).trace());
                }
                if constexpr(ENABLE_AD) {
                    op.obs_d1(x, y) = o_d1(x, y).val();
                } else {
                    op.obs_d1(x, y) = o_d1(x, y);
                }
            }
            if((bond & Opts::Bond::D2) == Opts::Bond::D2) {
                if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                    o_d2(x, y) = rho.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_d2(x, y)).trace();
                } else {
                    o_d2(x, y) = std::real(
                        rho.twist(0).twist(1).template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(op.data_d2(x, y)).trace());
                }
                if constexpr(ENABLE_AD) {
                    op.obs_d2(x, y) = o_d2(x, y).val();
                } else {
                    op.obs_d2(x, y) = o_d2(x, y);
                }
            }
        }
    }
    return std::array{o_h, o_v, o_d1, o_d2};
}

template <typename Scalar,
          typename Symmetry,
          std::size_t TRank,
          bool ALL_OUT_LEGS,
          bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts,
          typename OpScalar,
          bool HERMITIAN>
TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename OneSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>
avg(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env, OneSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    using ObsScalar = typename OneSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar;
    // Energy-only solves do not allocate RDMs. Materialize a measurement
    // environment on demand, preserving the caller's const environment.
    if(!env.RDM_COMPUTED()) {
        auto measured = env;
        measured.template computeRDM<ENABLE_AD>();
        return avg(measured, op);
    }
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o(env.cell().pattern);
    o.setConstant(0.);
    if(not op.MEASURE) { return o; }

    auto shifted_op = op.shiftQN(env.Psi()->charges());

    for(int x = 0; x < env.cell().rows(); ++x) {
        for(int y = 0; y < env.cell().cols(); ++y) {
            if(not env.cell().pattern.isUnique(x, y)) { continue; }
            // auto C1T1 = env.C1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1,
            // ENABLE_AD>(env.T1s(x, y - 1)); auto T4C1T1 = env.T4s(x - 1, y).template contract<std::array{1, -1, -2, -3},
            // std::array{1, -4, -5, -6}, 3, ENABLE_AD>(C1T1); auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2,
            // -4}, std::array{1, 2, -5, -6, -7}, 4, ENABLE_AD>(env.A->As(x, y)); auto T4C1T1AH = T4C1T1A.template
            // contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{1, -7}, 6, ENABLE_AD>(op.data(x, y)); auto Q1H =
            // T4C1T1AH.template contract<std::array{-1, 1, -4, 2, -5, -2, 3}, std::array{1, 2, 3, -6, -3}, 3,
            // ENABLE_AD>(env.A->Adags(x, y));

            // auto C2T2 = env.C2s(x + 1, y - 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 3>(env.T2s(x + 1,
            // y)); auto C2T2C3 = C2T2.template contract<std::array{-1, -2, -3, 1}, std::array{1, -1}, 3>(env.C3s(x + 1, y + 1)); auto
            // C2T2C3T3 = C2T2C3.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, -6, 1}, 3>(env.T3s(x, y + 1)); auto
            // C2T2C3T3C4 = C2T2C3T3.template contract<std::array{-1, -2, -3, -5, -6, 1}, std::array{-4, 1}, 3>(env.C4s(x - 1, y +
            // 1)); auto Q1 = env.contractCorner(x, y, Opts::CORNER::UPPER_LEFT);
            // // auto norm = (C2T2C3T3C4 * Q1).trace();
            // // o(x, y) = (C2T2C3T3C4 * Q1H).trace() / norm;
            // auto norm = (Q1 * C2T2C3T3C4).trace();
            // o(x, y) = (Q1H * C2T2C3T3C4).trace() / norm;
            if constexpr(ALL_OUT_LEGS) {
                if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                    o(x, y) =
                        env.rho1_h(x, y).twist(0).template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(shifted_op.data(x, y)).trace();
                } else {
                    o(x, y) = std::real(
                        env.rho1_h(x, y).twist(0).template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(shifted_op.data(x, y)).trace());
                }
            } else {
                if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                    o(x, y) = 0.5 * (env.rho1_h(x, y)
                                         .twist(0)
                                         .template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(shifted_op.data(x, y))
                                         .trace() +
                                     env.rho1_v(x, y)
                                         .twist(0)
                                         .template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(shifted_op.data(x, y))
                                         .trace());
                } else {
                    o(x, y) = 0.5 * std::real((env.rho1_h(x, y)
                                                   .twist(0)
                                                   .template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(shifted_op.data(x, y))
                                                   .trace() +
                                               env.rho1_v(x, y)
                                                   .twist(0)
                                                   .template contract<std::array{1, 2}, std::array{2, 1}, 0, ENABLE_AD>(shifted_op.data(x, y))
                                                   .trace()));
                }
            }
            if constexpr(ENABLE_AD) {
                op.obs(x, y) = o(x, y).val();
            } else {
                op.obs(x, y) = o(x, y);
            }
        }
    }
    return o;
}

template <typename Scalar,
          typename Symmetry,
          std::size_t TRank,
          bool ALL_OUT_LEGS,
          bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts,
          typename OpScalar,
          bool HERMITIAN>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>, 4>
avg_unsliced(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op, bool preserve_operator_multiplets)
{
    assert(env.RDM_COMPUTED());

    using ObsScalar = typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar;

    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o_h(env.cell().pattern);
    o_h.setConstant(0.);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o_v(env.cell().pattern);
    o_v.setConstant(0.);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o_d1(env.cell().pattern);
    o_d1.setConstant(0.);
    TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, ObsScalar>> o_d2(env.cell().pattern);
    o_d2.setConstant(0.);
    if(not op.MEASURE) { return std::array{o_h, o_v, o_d1, o_d2}; }

    auto shifted_op = op.shiftQN(env.Psi()->charges());

    for(int x = 0; x < env.cell().rows(); ++x) {
        for(int y = 0; y < env.cell().cols(); ++y) {
            if(not env.cell().pattern.isUnique(x, y)) { continue; }
            if(op.data_h.size() > 0) {
                if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                    o_h(x, y) = env.rho_h(x, y)
                                    .twist(0)
                                    .twist(1)
                                    .template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(shifted_op.data_h(x, y))
                                    .trace();
                } else {
                    o_h(x, y) = std::real(env.rho_h(x, y)
                                              .twist(0)
                                              .twist(1)
                                              .template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(shifted_op.data_h(x, y))
                                              .trace());
                }
            }
            if(op.data_v.size() > 0) {
                if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                    o_v(x, y) = env.rho_v(x, y)
                                    .twist(0)
                                    .twist(1)
                                    .template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(shifted_op.data_v(x, y))
                                    .trace();
                } else {
                    o_v(x, y) = std::real(env.rho_v(x, y)
                                              .twist(0)
                                              .twist(1)
                                              .template contract<std::array{1, 2, 3, 4}, std::array{3, 4, 1, 2}, 0>(shifted_op.data_v(x, y))
                                              .trace());
                }
            }
            if constexpr(ENABLE_AD) {
                if(op.data_h.size() > 0) { op.obs_h(x, y) = o_h(x, y).val(); }
                if(op.data_v.size() > 0) { op.obs_v(x, y) = o_v(x, y).val(); }
            } else {
                if(op.data_h.size() > 0) { op.obs_h(x, y) = o_h(x, y); }
                if(op.data_v.size() > 0) { op.obs_v(x, y) = o_v(x, y); }
            }

            if(op.data_d1.size() > 0 or op.data_d2.size() > 0) {
                if constexpr(ALL_OUT_LEGS) {
                    auto Q = env.contractCorner(0, 0, Opts::CORNER::UPPER_LEFT);
                    auto half = Q * Q.adjoint();
                    auto norm = (half * half.adjoint()).trace();
                    [[maybe_unused]] double dumb;
                    auto [Hu, Hs, Hvdag] =
                        shifted_op.data_d1(x, y).template permute<0, 0, 2, 1, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, preserve_operator_multiplets);
                    Hu = Hu * Hs.diag_sqrt().eval();
                    Hvdag = Hs.diag_sqrt().eval() * Hvdag;
                    auto C1T1 =
                        env.C1s(0, 0).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1, ENABLE_AD>(env.T1s(0, 0).adjoint().eval());
                    auto T4C1T1 =
                        env.T1s(0, 0).adjoint().eval().template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3, ENABLE_AD>(
                            C1T1.template twist<ENABLE_AD>(0));
                    auto T4C1T1A =
                        T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, ENABLE_AD>(env.A->As(0, 0));
                    auto T4C1T1AHu = T4C1T1A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{1, -7, -8}, 6, ENABLE_AD>(Hu);
                    auto Q1h = T4C1T1AHu.template contract<std::array{-1, 1, -5, 2, -6, -2, 3, -4}, std::array{1, 2, -7, -3, 3}, 4, ENABLE_AD>(
                        env.A->Adags(0, 0).twist(3).twist(4));
                    auto T4C1T1AHvdag = T4C1T1A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{-8, 1, -7}, 3, ENABLE_AD>(Hvdag);
                    auto Q3h = T4C1T1AHvdag.template contract<std::array{-1, 1, -5, 2, -6, -2, 3, -4}, std::array{1, 2, -7, -3, 3}, 4, ENABLE_AD>(
                        env.A->Adags(0, 0).twist(3).twist(4));
                    auto upper_half = Q1h * Q.adjoint();
                    auto lower_half = Q3h * Q.adjoint();
                    if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                        o_d1(x, y) =
                            upper_half.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{5, 6, 7, 4, 1, 2, 3}, 0, ENABLE_AD>(lower_half)
                                .trace() /
                            norm;
                    } else {
                        o_d1(x, y) = std::real(
                            upper_half.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{5, 6, 7, 4, 1, 2, 3}, 0, ENABLE_AD>(lower_half)
                                .trace() /
                            norm);
                    }
                    if constexpr(ENABLE_AD) {
                        op.obs_d1(x, y) = o_d1(x, y).val();
                    } else {
                        op.obs_d1(x, y) = o_d1(x, y);
                    }
                } else {
                    auto Q1 = env.contractCorner(x, y, Opts::CORNER::UPPER_LEFT);
                    auto Q2 = env.contractCorner(x + 1, y, Opts::CORNER::UPPER_RIGHT);
                    auto Q3 = env.contractCorner(x + 1, y + 1, Opts::CORNER::LOWER_RIGHT);
                    auto Q4 = env.contractCorner(x, y + 1, Opts::CORNER::LOWER_LEFT);
                    auto norm = (Q1 * Q2 * Q3 * Q4.twist(0).twist(1).twist(2)).trace();

                    if(op.data_d1.size() > 0) {
                        if constexpr(TRank == 2) {
                            [[maybe_unused]] double dumb;
                            auto [Hu, Hs, Hvdag] =
                                shifted_op.data_d1(x, y).template permute<0, 0, 2, 1, 3>().tSVD(std::numeric_limits<std::size_t>::max(), 0., dumb, preserve_operator_multiplets);
                            Hu = Hu * Hs.diag_sqrt().eval();
                            Hvdag = Hs.diag_sqrt().eval() * Hvdag;

                            auto C1T1 = env.C1s(x - 1, y - 1)
                                            .template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1, ENABLE_AD>(env.T1s(x, y - 1));
                            auto T4C1T1 = env.T4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3, ENABLE_AD>(
                                C1T1.twist(0));
                            auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, ENABLE_AD>(
                                env.A->As(x, y));
                            auto T4C1T1AH = T4C1T1A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{1, -7, -8}, 6, ENABLE_AD>(Hu);
                            auto Q1H = T4C1T1AH.template contract<std::array{-1, 1, -4, 2, -5, -2, 3, -7}, std::array{1, 2, 3, -6, -3}, 3, ENABLE_AD>(
                                env.A->Adags(x, y).twist(3).twist(4));

                            auto C3T3 = env.C3s(x + 2, y + 2)
                                            .twist(1)
                                            .template contract<std::array{-1, 1}, std::array{-2, -3, -4, 1}, 1, ENABLE_AD>(env.T3s(x + 1, y + 2));
                            auto T2C3T3 =
                                env.T2s(x + 2, y + 1).template contract<std::array{-1, -2, -3, 1}, std::array{1, -4, -5, -6}, 3, ENABLE_AD>(C3T3);
                            auto T2C3T3A = T2C3T3.template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, -6, 1, 2, -7}, 4, ENABLE_AD>(
                                env.A->As(x + 1, y + 1).twist(2).twist(3));
                            auto T2C3T3AH =
                                T2C3T3A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{-7, 1, -8}, 6, ENABLE_AD>(Hvdag);
                            auto Q3H = T2C3T3AH.template contract<std::array{1, -1, 2, -4, -5, -2, -7, 3}, std::array{-6, -3, 3, 1, 2}, 3, ENABLE_AD>(
                                env.A->Adags(x + 1, y + 1));
                            Q1H = Q4.twist(3).twist(4).twist(5) * Q1H;
                            Q3H = Q2 * Q3H;
                            if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                                o_d1(x, y) =
                                    Q1H.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{4, 5, 6, 1, 2, 3, 7}, 0, ENABLE_AD>(Q3H)
                                        .trace() /
                                    norm;
                            } else {
                                o_d1(x, y) = std::real(
                                    Q1H.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{4, 5, 6, 1, 2, 3, 7}, 0, ENABLE_AD>(Q3H)
                                        .trace() /
                                    norm);
                            }
                        } else if constexpr(TRank == 1) {
                            o_d1(x, y) = 0.;
                        }
                        if constexpr(ENABLE_AD) {
                            op.obs_d1(x, y) = o_d1(x, y).val();
                        } else {
                            op.obs_d1(x, y) = o_d1(x, y);
                        }
                    }
                    if(op.data_d2.size() > 0) {
                        // Q1 = env.contractCorner(x - 1, y, Opts::CORNER::UPPER_LEFT);
                        // Q2 = env.contractCorner(x, y, Opts::CORNER::UPPER_RIGHT);
                        // Q3 = env.contractCorner(x, y + 1, Opts::CORNER::LOWER_RIGHT);
                        // Q4 = env.contractCorner(x - 1, y + 1, Opts::CORNER::LOWER_LEFT);
                        // norm = (Q1 * Q2 * Q3 * Q4).trace();
                        if constexpr(TRank == 2) {
                            [[maybe_unused]] double dumb;
                            auto [Hu, Hs, Hvdag] = shifted_op.data_d2(x + 1, y).template permute<0, 0, 2, 1, 3>().tSVD(
                                std::numeric_limits<std::size_t>::max(), 0., dumb, preserve_operator_multiplets);
                            Hu = Hu * Hs.diag_sqrt().eval();
                            Hvdag = Hs.diag_sqrt().eval() * Hvdag;

                            auto T1C2 = env.T1s(x + 1, y - 1)
                                            .template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 3, ENABLE_AD>(env.C2s(x + 2, y - 1));
                            auto T1C2T2 =
                                T1C2.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, 1, -6}, 3, ENABLE_AD>(env.T2s(x + 2, y));
                            auto T1C2T2A = T1C2T2.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{-5, 1, 2, -6, -7}, 4, ENABLE_AD>(
                                env.A->As(x + 1, y).twist(2));
                            auto T1C2T2AH = T1C2T2A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{1, -8, -7}, 6, ENABLE_AD>(Hu);
                            auto Q2H = T1C2T2AH.template contract<std::array{-1, 1, 2, -4, -2, -5, -7, 3}, std::array{-3, 1, 3, 2, -6}, 3, ENABLE_AD>(
                                env.A->Adags(x + 1, y).twist(4));

                            auto C4T3 = env.C4s(x - 1, y + 2)
                                            .template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1, ENABLE_AD>(env.T3s(x, y + 2));
                            auto T4C4T3 =
                                env.T4s(x - 1, y + 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3, ENABLE_AD>(C4T3);
                            auto T4C4T3A = T4C4T3.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{1, -5, -6, 2, -7}, 4, ENABLE_AD>(
                                env.A->As(x, y + 1).twist(3));
                            auto T4C4T3AH =
                                T4C4T3A.template contract<std::array{-1, -2, -3, -4, -5, -6, 1}, std::array{-8, 1, -7}, 6, ENABLE_AD>(Hvdag);
                            auto Q4H = T4C4T3AH.template contract<std::array{-4, 1, 2, -1, -5, -2, 3, -7}, std::array{1, -6, 3, -3, 2}, 3, ENABLE_AD>(
                                env.A->Adags(x, y + 1).twist(3));
                            Q2H = Q1 * Q2H;
                            Q4H = Q3 * Q4H.twist(0).twist(1).twist(2);
                            if constexpr(ScalarTraits<ObsScalar>::IS_COMPLEX()) {
                                o_d2(x, y) =
                                    Q2H.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{4, 5, 6, 1, 2, 3, 7}, 0, ENABLE_AD>(Q4H)
                                        .trace() /
                                    norm;
                            } else {
                                o_d2(x, y) = std::real(
                                    Q2H.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{4, 5, 6, 1, 2, 3, 7}, 0, ENABLE_AD>(Q4H)
                                        .trace() /
                                    norm);
                            }
                        } else if constexpr(TRank == 1) {
                            o_d2(x, y) = 0.;
                        }
                        // o_d2(x, y) =
                        //     Q2H.template contract<std::array{1, 2, 3, 4, 5, 6, 7}, std::array{1, 2, 3, 4, 5, 6, 7}, 0,
                        //     ENABLE_AD>(Q4H).trace() / norm;
                        if constexpr(ENABLE_AD) {
                            op.obs_d2(x, y) = o_d2(x, y).val();
                        } else {
                            op.obs_d2(x, y) = o_d2(x, y);
                        }
                    }
                }
            }
        }
    }
    return std::array{o_h, o_v, o_d1, o_d2};
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts, typename OpScalar, bool HERMITIAN>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>, 4>
avg_sliced(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env,
           TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op, std::size_t block_size)
{
    if(block_size == 0) { throw std::invalid_argument("Energy slice block size must be positive"); }
    if constexpr(detail::supports_sliced_energy<Scalar, Symmetry, TRank, ALL_OUT_LEGS, OpScalar>) {
#ifdef XPED_USE_EIGEN_MATRIX_LIB
        using Result = std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>;
        TMatrix<Result> empty(env.cell().pattern);
        empty.setConstant(0.);
        std::array<TMatrix<Result>, 4> result{empty, empty, empty, empty};
        if(!op.MEASURE) { return result; }
        auto shifted = op.shiftQN(env.Psi()->charges());
        for(int x = 0; x < env.cell().rows(); ++x) {
            for(int y = 0; y < env.cell().cols(); ++y) {
                if(!env.cell().pattern.isUnique(x, y)) { continue; }
                auto evaluate = [&](const auto& term, auto& observations, Opts::Bond bond, std::size_t index, int ox) {
                    if(term.size() == 0) { return; }
                    auto value = detail::sliced_energy<ENABLE_AD>(env, term(ox, y), x, y, bond, block_size);
                    result[index](x, y) = value;
                    if constexpr(ENABLE_AD) { observations(x, y) = value.val(); }
                    else { observations(x, y) = value; }
                };
                evaluate(shifted.data_h, op.obs_h, Opts::Bond::H, 0, x);
                evaluate(shifted.data_v, op.obs_v, Opts::Bond::V, 1, x);
                evaluate(shifted.data_d1, op.obs_d1, Opts::Bond::D1, 2, x);
                // Preserve the existing D2 anchoring at the upper-right site.
                evaluate(shifted.data_d2, op.obs_d2, Opts::Bond::D2, 3, x + 1);
            }
        }
        return result;
#endif
    } else {
        if(env.RDM_COMPUTED()) { return avg_unsliced(env, op); }
        auto measured = env;
        measured.template computeRDM<ENABLE_AD>();
        return avg_unsliced(measured, op);
    }
}

template <typename Scalar, typename Symmetry, std::size_t TRank, bool ALL_OUT_LEGS, bool ENABLE_AD,
          Opts::CTMCheckpoint CPOpts, typename OpScalar, bool HERMITIAN>
std::array<TMatrix<std::conditional_t<ENABLE_AD, stan::math::var, typename TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>::ObsScalar>>, 4>
avg(XPED_CONST CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts>& env, TwoSiteObservable<OpScalar, Symmetry, HERMITIAN>& op)
{
    return avg_sliced(env, op);
}

} // namespace Xped

#if __has_include("LinearAlgebra.gen.cpp")
#    include "LinearAlgebra.gen.cpp"
#endif
