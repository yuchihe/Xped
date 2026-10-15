// Standalone regression for free-leg corner blocking and its native AD replay.
#include <cmath>
#include <cstdio>
#include <random>

#include "stan/math/rev.hpp"
#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"
#include "Xped/Core/Qbasis.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/AD/ADTensor.hpp"
#include "Xped/PEPS/CornerBlocking.hpp"

static int failures = 0;
#define CHECK(condition) do { if(!(condition)) { std::printf("FAIL line %d: %s\n", __LINE__, #condition); ++failures; } } while(false)

template <typename T>
double relative_error(const T& a, const T& b)
{
    return (a - b).norm() / (1. + b.norm());
}

template <typename T>
double directional_dot(const T& gradient, const T& direction)
{
    // Reduced block adjoints use the same metric as Xped's tensor AD rules.
    double result = 0.;
    for(std::size_t i = 0; i < gradient.dict().size(); ++i) {
        result += (gradient.block(i).array() * direction.block(i).array()).sum();
    }
    return result;
}

template <typename Symmetry, std::size_t Leg>
void run_case(const char* name, const Xped::Qbasis<Symmetry, 1>& b1, const Xped::Qbasis<Symmetry, 1>& b2)
{
    using Plain = Xped::Tensor<double, 2, 1, Symmetry, false>;
    using Active = Xped::Tensor<double, 2, 1, Symmetry, true>;
    const auto combined = b1.combine(b2).forgetHistory();
    using Shared = std::conditional_t<Leg == 2, Xped::Tensor<double, 2, 2, Symmetry, false>,
                                              Xped::Tensor<double, 1, 1, Symmetry, false>>;
    using SharedAD = std::conditional_t<Leg == 2, Xped::Tensor<double, 2, 2, Symmetry, true>,
                                                Xped::Tensor<double, 1, 1, Symmetry, true>>;
    Plain input({{b1, b2}}, {{combined}});
    Shared shared;
    if constexpr(Leg == 2) shared = Shared({{b1, b2}}, {{b1, b2}});
    else shared = Shared({{combined}}, {{combined}});
    std::mt19937 random(42 + Leg);
    input.setRandom(random);
    shared.setRandom(random);
    auto input_direction = input;
    auto shared_direction = shared;
    input_direction.setRandom(random);
    shared_direction.setRandom(random);

    auto contraction = []<bool Track>(const auto& edge, const auto& factor) {
        if constexpr(Leg == 2) {
            return factor.template contract<std::array{-1, -2, 1, 2}, std::array{1, 2, -3}, 2, Track>(edge);
        } else {
            auto product = edge.template contract<std::array{-1, -2, 1}, std::array{1, -3}, 2, Track>(factor);
            if constexpr(Leg == 0) return product.template permute<1, 1, 2, 0>(Xped::Bool<Track>{});
            else return product;
        }
    };
    constexpr std::size_t OutLeg = Leg == 0 ? 2 : Leg;
    const auto expected = contraction.template operator()<false>(input, shared);
    Plain reference_input;
    Shared reference_shared;
    {
        stan::math::nested_rev_autodiff nested;
        Active edge(input);
        SharedAD factor(shared);
        auto output = contraction.template operator()<true>(edge, factor);
        auto loss = output.norm();
        stan::math::grad(loss.vi_);
        reference_input = edge.adj();
        reference_shared = factor.adj();
    }

    for(const std::size_t width : {1ul, 3ul, 8ul, 100ul}) {
        auto plain_fn = [&]<bool Track>(const auto& edge) { return contraction.template operator()<Track>(edge, shared); };
        auto plain_output = Xped::detail::blocked_corner<Leg, OutLeg, false>(input, plain_fn, width);
        CHECK(relative_error(plain_output, expected) < 2.e-12);
        stan::math::nested_rev_autodiff nested;
        Active edge(input);
        SharedAD factor(shared);
        auto fn = [factor, contraction]<bool Track>(const auto& slice) { return contraction.template operator()<Track>(slice, factor); };
        const auto before = stan::math::ChainableStack::instance_->var_stack_.size();
        auto output = Xped::detail::blocked_corner<Leg, OutLeg, true>(edge, fn, width);
        const auto recorded = stan::math::ChainableStack::instance_->var_stack_.size() - before;
        CHECK(recorded == 1); // One replay callback, independent of slice count.
        CHECK(relative_error(output.val(), expected) < 2.e-12);
        auto loss = output.norm();
        stan::math::grad(loss.vi_);
        CHECK(relative_error(edge.adj(), reference_input) < 2.e-11);
        CHECK(relative_error(factor.adj(), reference_shared) < 2.e-11);
        // Independent directional finite differences catch a shared error in AD paths.
        constexpr double h = 1.e-5;
        Plain plus_input = input + h * input_direction;
        Plain minus_input = input - h * input_direction;
        Shared plus_shared = shared + h * shared_direction;
        Shared minus_shared = shared - h * shared_direction;
        const double fd = (contraction.template operator()<false>(plus_input, plus_shared).norm()
                         - contraction.template operator()<false>(minus_input, minus_shared).norm()) / (2. * h);
        const double ad = directional_dot(edge.adj(), input_direction) + directional_dot(factor.adj(), shared_direction);
        CHECK(std::abs(ad - fd) / (1. + std::abs(fd)) < 2.e-7);
        std::printf("%s leg=%zu width=%zu: input %.3e shared %.3e FD %.3e nodes=%zu\n", name, Leg, width,
                    relative_error(edge.adj(), reference_input), relative_error(factor.adj(), reference_shared),
                    std::abs(ad - fd) / (1. + std::abs(fd)), recorded);
    }
}

template <typename Symmetry>
void run_symmetry(const char* name, const Xped::Qbasis<Symmetry, 1>& b1, const Xped::Qbasis<Symmetry, 1>& b2)
{
    run_case<Symmetry, 0>(name, b1, b2);
    run_case<Symmetry, 1>(name, b1, b2);
    run_case<Symmetry, 2>(name, b1, b2);
}

int main()
{
    Xped::Qbasis<Xped::Sym::U0<>, 1> u0a, u0b;
    u0a.push_back({}, 11); u0b.push_back({}, 3);
    run_symmetry("U0", u0a, u0b);
    using U1 = Xped::Sym::U1<Xped::Sym::SpinU1>;
    Xped::Qbasis<U1, 1> u1a, u1b;
    u1a.push_back({-1}, 5); u1a.push_back({0}, 3); u1a.push_back({1}, 2);
    u1b.push_back({-1}, 2); u1b.push_back({1}, 1);
    run_symmetry("U1", u1a, u1b);
    using SU2 = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
    Xped::Qbasis<SU2, 1> su2a, su2b;
    su2a.push_back({1}, 5); su2a.push_back({2}, 3); su2a.push_back({3}, 1);
    su2b.push_back({1}, 2); su2b.push_back({2}, 1);
    run_symmetry("SU2", su2a, su2b);
    std::printf("Corner blocking: %d failures\n", failures);
    return failures ? 1 : 0;
}
