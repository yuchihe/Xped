// Compare every corner value and input adjoint against the original unblocked contractions.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <random>
#include "stan/math/rev.hpp"
#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Core/AdjointOp.hpp"
#include "Xped/PEPS/CTM.hpp"

using Symmetry = Xped::Sym::U0<>;
using Basis = Xped::Qbasis<Symmetry, 1>;

template <std::size_t Rank, std::size_t CoRank>
auto random_tensor(const std::array<Basis, Rank>& domain, const std::array<Basis, CoRank>& codomain, std::mt19937& random)
{
    Xped::Tensor<double, Rank, CoRank, Symmetry, false> tensor(domain, codomain);
    tensor.setRandom(random);
    return tensor;
}

template <typename Matrix, typename T>
void put(Matrix& matrix, const Xped::Pattern& pattern, const T& tensor)
{
    matrix.resize(pattern);
    matrix[0] = tensor;
}

template <std::size_t TRank, bool AllOut, bool Track>
void fixture(Xped::CTM<double, Symmetry, TRank, AllOut, Track>& env)
{
    env.cell_ = Xped::UnitCell(1, 1);
    env.chi_ = 11;
    const auto& pattern = env.cell_.pattern;
    Basis chi, aux, phys, fused;
    chi.push_back({}, 11); aux.push_back({}, 2); phys.push_back({}, 3); fused.push_back({}, 4);
    std::mt19937 random(738);
    if constexpr(AllOut) {
        put(env.C1s, pattern, random_tensor<2, 0>({chi, chi}, {}, random));
        put(env.T1s, pattern, random_tensor<4, 0>({chi, chi, aux, aux}, {}, random));
    } else {
        put(env.C1s, pattern, random_tensor<0, 2>({}, {chi, chi}, random));
        put(env.C2s, pattern, random_tensor<1, 1>({chi}, {chi}, random));
        put(env.C3s, pattern, random_tensor<2, 0>({chi, chi}, {}, random));
        put(env.C4s, pattern, random_tensor<1, 1>({chi}, {chi}, random));
        if constexpr(TRank == 2) {
            put(env.T1s, pattern, random_tensor<1, 3>({chi}, {chi, aux, aux}, random));
            put(env.T2s, pattern, random_tensor<3, 1>({aux, aux, chi}, {chi}, random));
            put(env.T3s, pattern, random_tensor<3, 1>({aux, aux, chi}, {chi}, random));
            put(env.T4s, pattern, random_tensor<1, 3>({chi}, {chi, aux, aux}, random));
        } else {
            put(env.T1s, pattern, random_tensor<1, 2>({chi}, {chi, fused}, random));
            put(env.T2s, pattern, random_tensor<2, 1>({fused, chi}, {chi}, random));
            put(env.T3s, pattern, random_tensor<2, 1>({fused, chi}, {chi}, random));
            put(env.T4s, pattern, random_tensor<1, 2>({chi}, {chi, fused}, random));
            put(env.Ms, pattern, random_tensor<2, 2>({fused, fused}, {fused, fused}, random));
        }
    }
    if constexpr(TRank == 2) {
        env.A = std::make_shared<Xped::iPEPS<double, Symmetry, AllOut, Track>>();
        if constexpr(AllOut) {
            put(env.A->As, pattern, random_tensor<4, 1>({aux, aux, aux, aux}, {phys}, random));
            put(env.A->Adags, pattern, random_tensor<5, 0>({aux, aux, aux, aux, phys}, {}, random));
        } else {
            put(env.A->As, pattern, random_tensor<2, 3>({aux, aux}, {aux, aux, phys}, random));
            put(env.A->Adags, pattern, random_tensor<3, 2>({aux, aux, phys}, {aux, aux}, random));
        }
    }
}

// Preserve the pre-blocking contraction order as an independent integration reference.
// The checkpoint wrapper is intentionally absent: compare against the full AD graph.
template <std::size_t TRank, bool AllOut, bool TRACK>
Xped::Tensor<double, TRank + 1, TRank + 1, Symmetry, TRACK>
unblocked(Xped::CTM<double, Symmetry, TRank, AllOut, TRACK>& env, Xped::Opts::CORNER corner)
{
    constexpr int x = 0, y = 0;
    constexpr bool TRACK_INNER = TRACK;

    Xped::Tensor<double, TRank + 1, TRank + 1, Symmetry, TRACK> Q;
    if constexpr(AllOut) {
        auto C1T1 = env.C1s(0, 0).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1, TRACK_INNER>(env.T1s(0, 0).adjoint().eval());
        auto T4C1T1 = env.T1s(0, 0).adjoint().eval().template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(
            C1T1.template twist<TRACK_INNER>(0));
        auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, TRACK_INNER>(env.A->As(0, 0));
        Q = T4C1T1A.template contract<std::array{-1, 1, -4, 2, -5, -2, 3}, std::array{1, 2, -6, -3, 3}, 3, TRACK_INNER>(
            env.A->Adags(0, 0).template twist<TRACK_INNER>(3).template twist<TRACK_INNER>(4));
    } else {
        switch(corner) {
        case Xped::Opts::CORNER::UPPER_LEFT: {
            // ooooo -->--
            // o Q o
            // ooooo ==>==
            // |  ||
            // ^  ^^
            // |  ||
            if constexpr(TRank == 2) {
                auto C1T1 = env.C1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3, -4}, 1, TRACK_INNER>(env.T1s(x, y - 1));
                auto T4C1T1 = env.T4s(x - 1, y).template contract<std::array{1, -1, -2, -3}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(
                    C1T1.template twist<TRACK_INNER>(0));
                auto T4C1T1A = T4C1T1.template contract<std::array{-1, 1, -2, -3, 2, -4}, std::array{1, 2, -5, -6, -7}, 4, TRACK_INNER>(env.A->As(x, y));
                Q = T4C1T1A.template contract<std::array{-1, 1, -4, 2, -5, -2, 3}, std::array{1, 2, 3, -6, -3}, 3, TRACK_INNER>(
                    env.A->Adags(x, y).template twist<TRACK_INNER>(3).template twist<TRACK_INNER>(4));
            } else if constexpr(TRank == 1) {
                auto C1T1 = env.C1s(x - 1, y - 1).template contract<std::array{-1, 1}, std::array{1, -2, -3}, 1, TRACK_INNER>(env.T1s(x, y - 1));
                auto T4C1T1 = env.T4s(x - 1, y).template contract<std::array{1, -1, -2}, std::array{1, -4, -5}, 2, TRACK_INNER>(
                    C1T1.template twist<TRACK_INNER>(0));
                Q = T4C1T1.template contract<std::array{-1, 1, -3, 2}, std::array{1, 2, -4, -2}, 2>(env.Ms(x, y));
            }
            break;
        }
        case Xped::Opts::CORNER::LOWER_LEFT: {
            // |  ||
            // ^  ^^
            // |  ||
            // ooooo --<--
            // o Q o
            // ooooo ==<==
            if constexpr(TRank == 2) {
                auto C4T3 = env.C4s(x - 1, y + 1).template contract<std::array{-1, 1}, std::array{-2, -3, 1, -4}, 1, TRACK_INNER>(env.T3s(x, y + 1));
                auto T4C4T3 = env.T4s(x - 1, y).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(C4T3);
                auto T4C4T3A = T4C4T3.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{1, -5, -6, 2, -7}, 4, TRACK_INNER>(
                    env.A->As(x, y).template twist<TRACK_INNER>(3));
                Q = T4C4T3A.template contract<std::array{-4, 1, 2, -1, -5, -2, 3}, std::array{1, -6, 3, -3, 2}, 3, TRACK_INNER>(
                    env.A->Adags(x, y).template twist<TRACK_INNER>(3));
            } else if constexpr(TRank == 1) {
                auto C4T3 = env.C4s(x - 1, y + 1).template contract<std::array{-1, 1}, std::array{-2, 1, -3}, 1, TRACK_INNER>(env.T3s(x, y + 1));
                auto T4C4T3 = env.T4s(x - 1, y).template contract<std::array{-1, 1, -2}, std::array{1, -3, -4}, 2, TRACK_INNER>(C4T3);
                Q = T4C4T3.template contract<std::array{-3, 1, 2, -1}, std::array{1, -4, -2, 2}, 2>(env.Ms(x, y));
            }
            break;
        }
        case Xped::Opts::CORNER::UPPER_RIGHT: {
            // -->--ooooo
            //      o Q o
            // ==>==ooooo
            //      |  ||
            //      v  vv
            //      |  ||
            if constexpr(TRank == 2) {
                auto T1C2 = env.T1s(x, y - 1).template contract<std::array{-1, 1, -2, -3}, std::array{1, -4}, 3, TRACK_INNER>(env.C2s(x + 1, y - 1));
                auto T1C2T2 = T1C2.template contract<std::array{-1, -2, -3, 1}, std::array{-4, -5, 1, -6}, 3, TRACK_INNER>(env.T2s(x + 1, y));
                auto T1C2T2A = T1C2T2.template contract<std::array{-1, 1, -2, 2, -3, -4}, std::array{-5, 1, 2, -6, -7}, 4, TRACK_INNER>(
                    env.A->As(x, y).template twist<TRACK_INNER>(2));
                Q = T1C2T2A.template contract<std::array{-1, 1, 2, -4, -2, -5, 3}, std::array{-3, 1, 3, 2, -6}, 3, TRACK_INNER>(
                    env.A->Adags(x, y).template twist<TRACK_INNER>(4));
            } else if constexpr(TRank == 1) {
                auto T1C2 = env.T1s(x, y - 1).template contract<std::array{-1, 1, -2}, std::array{1, -3}, 2, TRACK_INNER>(env.C2s(x + 1, y - 1));
                auto T1C2T2 = T1C2.template contract<std::array{-1, -2, 1}, std::array{-3, 1, -4}, 2, TRACK_INNER>(env.T2s(x + 1, y));
                Q = T1C2T2.template contract<std::array{-1, 1, 2, -3}, std::array{-2, 1, 2, -4}, 2>(env.Ms(x, y));
            }
            break;
        }
        case Xped::Opts::CORNER::LOWER_RIGHT: {
            //      |  ||
            //      v  vv
            //      |  ||
            // --<--ooooo
            //      o Q o
            // ==<==ooooo
            if constexpr(TRank == 2) {
                auto C3T3 = env.C3s(x + 1, y + 1)
                                .template twist<TRACK_INNER>(1)
                                .template contract<std::array{-1, 1}, std::array{-2, -3, -4, 1}, 1, TRACK_INNER>(env.T3s(x, y + 1));
                auto T2C3T3 = env.T2s(x + 1, y).template contract<std::array{-1, -2, -3, 1}, std::array{1, -4, -5, -6}, 3, TRACK_INNER>(C3T3);
                auto T2C3T3A = T2C3T3.template contract<std::array{1, -1, -2, 2, -3, -4}, std::array{-5, -6, 1, 2, -7}, 4, TRACK_INNER>(
                    env.A->As(x, y).template twist<TRACK_INNER>(2).template twist<TRACK_INNER>(3));
                Q = T2C3T3A.template contract<std::array{1, -1, 2, -4, -5, -2, 3}, std::array{-6, -3, 3, 1, 2}, 3, TRACK_INNER>(env.A->Adags(x, y));
            } else if constexpr(TRank == 1) {
                auto C3T3 = env.C3s(x + 1, y + 1)
                                .template twist<TRACK_INNER>(1)
                                .template contract<std::array{-1, 1}, std::array{-2, -3, 1}, 1, TRACK_INNER>(env.T3s(x, y + 1));
                auto T2C3T3 = env.T2s(x + 1, y).template contract<std::array{-1, -2, 1}, std::array{1, -3, -4}, 2, TRACK_INNER>(C3T3);
                Q = T2C3T3.template contract<std::array{1, -1, 2, -3}, std::array{-4, -2, 1, 2}, 2>(env.Ms(x, y));
            }
            break;
        }
        }
    }

    return Q;
}

using Snapshot = std::vector<std::pair<std::string, std::vector<double>>>;
int failures = 0;

template <typename T>
void save(Snapshot& snapshot, const char* label, const T& tensor)
{
    auto& values = snapshot.emplace_back(label, std::vector<double>{}).second;
    values.reserve(tensor.plainSize());
    for(std::size_t i = 0; i < tensor.dict().size(); ++i) {
        const auto block = tensor.block(i);
        for(Eigen::Index column = 0; column < block.cols(); ++column)
            for(Eigen::Index row = 0; row < block.rows(); ++row) values.push_back(block(row, column));
    }
}

double compare(const Snapshot& actual, const Snapshot& expected)
{
    if(actual.size() != expected.size()) {
        ++failures;
        return 1.;
    }
    double error = 0.;
    for(std::size_t t = 0; t < actual.size(); ++t) {
        const auto& [label, values] = actual[t];
        const auto& reference = expected[t].second;
        if(label != expected[t].first || values.size() != reference.size()) {
            std::fprintf(stderr, "Mismatched tensor %s\n", label.c_str());
            ++failures;
            continue;
        }
        double scale = 1., difference = 0.;
        for(std::size_t i = 0; i < values.size(); ++i) {
            if(!std::isfinite(values[i]) || !std::isfinite(reference[i])) {
                std::fprintf(stderr, "Nonfinite tensor %s\n", label.c_str());
                ++failures;
                break;
            }
            scale = std::max(scale, std::abs(reference[i]));
            difference = std::max(difference, std::abs(values[i] - reference[i]));
        }
        const double relative = difference / scale;
        error = std::max(error, relative);
        if(relative > 2.e-12) {
            std::fprintf(stderr, "%s relative error %.3e\n", label.c_str(), relative);
            ++failures;
        }
    }
    return error;
}

template <std::size_t TRank, bool AllOut, bool CP, bool Blocked>
Snapshot evaluate_ad(Xped::Opts::CORNER corner)
{
    stan::math::nested_rev_autodiff nested;
    Xped::CTM<double, Symmetry, TRank, AllOut, true> env;
    fixture(env);
    const auto before = stan::math::ChainableStack::instance_->var_stack_.size();
    auto result = [&]() {
        if constexpr(Blocked) return env.template contractCorner<true, CP>(0, 0, corner);
        else return unblocked(env, corner);
    }();
    if constexpr(Blocked) {
        const auto recorded = stan::math::ChainableStack::instance_->var_stack_.size() - before;
        if(recorded != 1) {
            std::fprintf(stderr, "Expected one corner replay callback, got %zu\n", recorded);
            ++failures;
        }
    }
    auto loss = result.norm();
    stan::math::grad(loss.vi_);
    Snapshot snapshot;
    save(snapshot, "value", result.val());
    save(snapshot, "C1", env.C1s[0].adj()); save(snapshot, "T1", env.T1s[0].adj());
    if constexpr(!AllOut) {
        save(snapshot, "C2", env.C2s[0].adj()); save(snapshot, "C3", env.C3s[0].adj()); save(snapshot, "C4", env.C4s[0].adj());
        save(snapshot, "T2", env.T2s[0].adj()); save(snapshot, "T3", env.T3s[0].adj()); save(snapshot, "T4", env.T4s[0].adj());
    }
    if constexpr(TRank == 2) {
        save(snapshot, "A", env.A->As[0].adj()); save(snapshot, "Adag", env.A->Adags[0].adj());
    } else save(snapshot, "M", env.Ms[0].adj());
    return snapshot;
}

template <std::size_t TRank, bool AllOut, bool CP>
void run()
{
    // Build in place: CTM's copy constructor regenerates Ms from A, while this
    // small TRank=1 fixture intentionally supplies Ms directly.
    Xped::CTM<double, Symmetry, TRank, AllOut, false> plain;
    fixture(plain);
    for(auto corner : {Xped::Opts::CORNER::UPPER_LEFT, Xped::Opts::CORNER::UPPER_RIGHT,
                       Xped::Opts::CORNER::LOWER_LEFT, Xped::Opts::CORNER::LOWER_RIGHT}) {
        if constexpr(AllOut) { if(corner != Xped::Opts::CORNER::UPPER_LEFT) continue; }
        Snapshot actual, expected;
        save(actual, "value", plain.template contractCorner<false, CP>(0, 0, corner));
        save(expected, "value", unblocked(plain, corner));
        const double forward_error = compare(actual, expected);
        const auto reference_ad = evaluate_ad<TRank, AllOut, CP, false>(corner);
        const auto blocked_ad = evaluate_ad<TRank, AllOut, CP, true>(corner);
        const double adjoint_error = compare(blocked_ad, reference_ad);
        Snapshot ad_value{blocked_ad.front()};
        compare(ad_value, expected);
        std::printf("TRank=%zu all_out=%d checkpoint=%d corner=%d: forward %.3e AD %.3e\n",
                    TRank, AllOut, CP, int(corner), forward_error, adjoint_error);
    }
}

int main()
{
    run<2, false, false>(); run<2, false, true>();
    run<1, false, false>(); run<1, false, true>();
    run<2, true, false>(); run<2, true, true>();
    std::printf("CTM corner integration: %d failures\n", failures);
    return failures ? 1 : 0;
}
