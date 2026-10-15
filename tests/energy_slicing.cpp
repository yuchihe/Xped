// Exact NN/NNN observable regression against the original RDM/contraction path.
// All fixtures and perturbations are deterministic; no CTMRG iteration is used.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "stan/math/rev.hpp"
#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Core/AdjointOp.hpp"
#include "Xped/PEPS/LinearAlgebra.hpp"

namespace {
using Symmetry = Xped::Sym::U0<>;
using Basis = Xped::Qbasis<Symmetry, 1>;
template <bool AD> using Psi = Xped::iPEPS<double, Symmetry, false, AD>;
template <bool AD> using Env = Xped::CTM<double, Symmetry, 2, false, AD>;
using Observable = Xped::TwoSiteObservable<double, Symmetry>;
using SiteData = std::array<std::vector<double>, 9>; // A, C1..C4, T1..T4.
struct Inputs {
    std::size_t D = 2, physical = 2, Lx = 1, Ly = 1;
    std::array<std::size_t, 4> chi{2, 3, 2, 3}; // Top, right, bottom, left.
    std::vector<SiteData> sites;
};
struct Evaluation {
    double value = 0.;
    std::vector<SiteData> gradient;
};
enum class Path { Original, Sliced, Default };
int failures = 0, checks = 0;
double max_value_error = 0., max_gradient_error = 0., max_fd_error = 0.;

void check(bool condition, const std::string& label)
{
    ++checks;
    if(!condition) { ++failures; std::fprintf(stderr, "FAIL %s\n", label.c_str()); }
}

Basis basis(std::size_t size)
{
    Basis result;
    result.push_back({}, size);
    result.sort();
    return result;
}

Inputs fixture(std::size_t physical = 2, std::size_t Lx = 1, std::size_t Ly = 1, bool square = false)
{
    Inputs in;
    in.physical = physical; in.Lx = Lx; in.Ly = Ly;
    if(square) in.chi.fill(2);
    const auto [top, right, bottom, left] = in.chi;
    const std::array<std::size_t, 9> sizes{
        physical * in.D * in.D * in.D * in.D, left * top, top * right, right * bottom, left * bottom,
        top * top * in.D * in.D, right * right * in.D * in.D,
        bottom * bottom * in.D * in.D, left * left * in.D * in.D};
    in.sites.resize(Lx * Ly);
    for(std::size_t site = 0; site < in.sites.size(); ++site) {
        for(std::size_t group = 0; group < sizes.size(); ++group) {
            auto& values = in.sites[site][group];
            values.resize(sizes[group]);
            for(std::size_t i = 0; i < values.size(); ++i)
                values[i] = (1. + .23 * std::sin(.37 * (i + 1) + .83 * group + .61 * site)
                                  + .11 * std::cos(.19 * (i + 2) * (group + 1))) / std::sqrt(double(values.size()));
        }
    }
    return in;
}

template <std::size_t Rank, std::size_t CoRank>
auto tensor(const std::array<std::size_t, Rank>& rows, const std::array<std::size_t, CoRank>& cols,
            const std::vector<double>& values)
{
    std::array<Basis, Rank> domain;
    std::array<Basis, CoRank> codomain;
    for(std::size_t i = 0; i < Rank; ++i) domain[i] = basis(rows[i]);
    for(std::size_t i = 0; i < CoRank; ++i) codomain[i] = basis(cols[i]);
    Xped::Tensor<double, Rank, CoRank, Symmetry, false> result(domain, codomain);
    result.setZero();
    result.set_data(values.data(), values.size());
    return result;
}

template <bool AD>
std::unique_ptr<Env<AD>> environment(const Inputs& in, bool aliases = false)
{
    Xped::UnitCell cell(in.Lx, in.Ly);
    Xped::TMatrix<Basis> auxiliary(cell.pattern), physical(cell.pattern);
    auxiliary.setConstant(basis(in.D)); physical.setConstant(basis(in.physical));
    auto plain_psi = std::make_shared<Psi<false>>(cell, in.D, auxiliary, auxiliary, physical, Xped::Opts::DiscreteSym::None);
    for(std::size_t site = 0; site < in.sites.size(); ++site) {
        plain_psi->As[site].setZero();
        plain_psi->As[site].set_data(in.sites[site][0].data(), in.sites[site][0].size());
    }
    plain_psi->updateAdags();
    std::shared_ptr<Psi<AD>> psi;
    if constexpr(AD) {
        psi = std::make_shared<Psi<true>>(*plain_psi);
        // Both physical-layer contributions must reach the same active ket.
        psi->updateAdags();
    } else psi = plain_psi;
    auto env = std::make_unique<Env<AD>>(psi, *std::max_element(in.chi.begin(), in.chi.end()), Xped::Opts::CTM_INIT::FROM_TRIVIAL);
    env->init();
    const auto [top, right, bottom, left] = in.chi;
    for(std::size_t site = 0; site < in.sites.size(); ++site) {
        const auto& values = in.sites[site];
        env->C1s[site] = tensor<0, 2>({}, {left, top}, values[1]);
        env->C2s[site] = tensor<1, 1>({top}, {right}, values[2]);
        env->C3s[site] = tensor<2, 0>({right, bottom}, {}, values[3]);
        env->C4s[site] = tensor<1, 1>({left}, {bottom}, values[4]);
        env->T1s[site] = tensor<1, 3>({top}, {top, in.D, in.D}, values[5]);
        env->T2s[site] = tensor<3, 1>({in.D, in.D, right}, {right}, values[6]);
        env->T3s[site] = tensor<3, 1>({in.D, in.D, bottom}, {bottom}, values[7]);
        env->T4s[site] = tensor<1, 3>({left}, {left, in.D, in.D}, values[8]);
        if(aliases) {
            env->C4s[site] = env->C2s[site];
            env->T4s[site] = env->T1s[site];
            env->T3s[site] = env->T2s[site];
        }
    }
    env->HAS_RDM = false;
    return env;
}

Observable observable(const Inputs& in, int selected, int kind = 0)
{
    using Bond = Xped::Opts::Bond;
    const std::array<Bond, 4> bonds{Bond::H, Bond::V, Bond::D1, Bond::D2};
    const Bond enabled = selected < 0 ? Bond::H | Bond::V | Bond::D1 | Bond::D2 : bonds[selected];
    Xped::UnitCell cell(in.Lx, in.Ly);
    Observable result(cell.pattern, enabled, "deterministic oriented operator");
    std::array<Xped::TMatrix<Xped::Tensor<double, 2, 2, Symmetry, false>>*, 4> data{
        &result.data_h, &result.data_v, &result.data_d1, &result.data_d2};
    const std::size_t dim = in.physical * in.physical;
    for(int bond = 0; bond < 4; ++bond) {
        if(selected >= 0 && selected != bond) continue;
        for(std::size_t site = 0; site < in.sites.size(); ++site) {
            std::vector<double> values(dim * dim);
            for(std::size_t col = 0; col < dim; ++col)
                for(std::size_t row = 0; row < dim; ++row)
                    values[row + dim * col] = kind == 1 ? double(row == col) : kind == 2 ? 0.
                        : .17 * std::sin(.41 * (row + 1) + .67 * (col + 2) + .29 * bond + .13 * site)
                        + .09 * std::cos(.23 * (row + 2) * (col + 1)) + .03 * (row == col);
            (*data[bond])[site] = tensor<2, 2>({in.physical, in.physical}, {in.physical, in.physical}, values);
        }
    }
    return result;
}

template <bool AD>
auto objective(Env<AD>& env, Observable& op, Path path, std::size_t width)
{
    const auto terms = [&]() {
        if(path == Path::Original) {
            env.template computeRDM<AD>();
            // The legacy operator SVD trims its last multiplet by default.
            // Keep every singular value for an exact contraction reference.
            return Xped::avg_unsliced(env, op, false);
        }
        if(path == Path::Default) return Xped::avg(env, op);
        return Xped::avg_sliced(env, op, width);
    }();
    if(path != Path::Original) check(!env.RDM_COMPUTED(), "sliced observables do not materialize an RDM");
    auto result = terms[0].sum();
    // Unequal signs/weights prevent diagonal or orientation errors cancelling.
    result += .73 * terms[1].sum() - .41 * terms[2].sum() + 1.19 * terms[3].sum();
    return result;
}

template <typename Tensor>
std::vector<double> payload(const Tensor& value)
{
    const auto block = value.block(0);
    return std::vector<double>(block.data(), block.data() + block.size());
}

Evaluation evaluate(const Inputs& in, int selected, Path path, std::size_t width = 8, int kind = 0, bool aliases = false)
{
    Evaluation result;
    const auto* stack = stan::math::ChainableStack::instance_;
    const auto before = std::array{stack->var_stack_.size(), stack->var_nochain_stack_.size(), stack->var_alloc_stack_.size()};
    {
        stan::math::nested_rev_autodiff tape;
        auto env = environment<true>(in, aliases);
        auto op = observable(in, selected, kind);
        auto loss = objective(*env, op, path, width);
        result.value = loss.val();
        check(std::isfinite(result.value), "finite energy");
        stan::math::grad(loss.vi_);
        result.gradient.resize(in.sites.size());
        for(std::size_t site = 0; site < in.sites.size(); ++site) {
            result.gradient[site] = {payload(env->A->As[site].adj()), payload(env->C1s[site].adj()), payload(env->C2s[site].adj()),
                payload(env->C3s[site].adj()), payload(env->C4s[site].adj()), payload(env->T1s[site].adj()),
                payload(env->T2s[site].adj()), payload(env->T3s[site].adj()), payload(env->T4s[site].adj())};
        }
    }
    check(before == std::array{stack->var_stack_.size(), stack->var_nochain_stack_.size(), stack->var_alloc_stack_.size()}, "nested tape recovery");
    return result;
}

double plain_value(const Inputs& in, int selected, Path path, std::size_t width = 8, int kind = 0)
{
    auto env = environment<false>(in);
    auto op = observable(in, selected, kind);
    return objective(*env, op, path, width);
}

void compare(const Evaluation& actual, const Evaluation& reference, const std::string& label)
{
    const double value_error = std::abs(actual.value - reference.value) / std::max(1., std::abs(reference.value));
    max_value_error = std::max(max_value_error, value_error);
    check(value_error < 3.e-11, label + " energy");
    for(std::size_t site = 0; site < reference.gradient.size(); ++site) {
        for(std::size_t group = 0; group < 9; ++group) {
            const auto& a = actual.gradient[site][group];
            const auto& b = reference.gradient[site][group];
            check(a.size() == b.size(), label + " gradient shape");
            double error = 0., scale = 1.;
            for(std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
                check(std::isfinite(a[i]) && std::isfinite(b[i]), label + " finite gradient");
                error = std::max(error, std::abs(a[i] - b[i]));
                scale = std::max(scale, std::abs(b[i]));
            }
            max_gradient_error = std::max(max_gradient_error, error / scale);
            check(error / scale < 5.e-10, label + " site " + std::to_string(site) + " gradient " + std::to_string(group));
        }
    }
}

void finite_difference(const Inputs& in, int selected, const Evaluation& analytic)
{
    constexpr double h = 1.e-5;
    for(std::size_t group = 0; group < 9; ++group) {
        Inputs plus = in, minus = in;
        double derivative = 0., scale_derivative = 0.;
        for(std::size_t site = 0; site < in.sites.size(); ++site) {
            const auto& values = in.sites[site][group];
            for(std::size_t i = 0; i < values.size(); ++i) {
                const double direction = std::sin(.31 * (i + 1) + .47 * group + .59 * site) / std::sqrt(double(values.size()));
                plus.sites[site][group][i] += h * direction;
                minus.sites[site][group][i] -= h * direction;
                derivative += analytic.gradient[site][group][i] * direction;
                scale_derivative += analytic.gradient[site][group][i] * values[i];
            }
        }
        const double fd = (plain_value(plus, selected, Path::Original) - plain_value(minus, selected, Path::Original)) / (2. * h);
        const double error = std::abs(fd - derivative) / std::max({1., std::abs(fd), std::abs(derivative)});
        max_fd_error = std::max(max_fd_error, error);
        check(error < 2.e-7, "independent FD bond " + std::to_string(selected) + " input group " + std::to_string(group));
        check(std::abs(scale_derivative) < 3.e-10, "normalization scale-null group " + std::to_string(group));
    }
}

void main_cases()
{
    const Inputs in = fixture();
    for(int selected = 0; selected < 4; ++selected) {
        const auto reference = evaluate(in, selected, Path::Original);
        for(std::size_t width : {1ul, 2ul, 8ul}) {
            const auto actual = evaluate(in, selected, Path::Sliced, width);
            compare(actual, reference, "bond " + std::to_string(selected) + " width " + std::to_string(width));
            check(std::abs(actual.value - plain_value(in, selected, Path::Sliced, width)) < 3.e-11, "plain/AD sliced value");
        }
        const auto actual = evaluate(in, selected, Path::Default);
        compare(actual, reference, "default dispatch bond " + std::to_string(selected));
        finite_difference(in, selected, actual);
        std::printf("bond=%d energy=%.15g checked all A/C/T adjoints and nine FD directions\n", selected, actual.value);
    }
    compare(evaluate(in, -1, Path::Sliced, 2), evaluate(in, -1, Path::Original), "combined NN+NNN");
}

void extended_cases()
{
    for(const auto& in : {fixture(3), fixture(2, 2, 1), fixture(2, 1, 2)}) {
        const auto reference = evaluate(in, -1, Path::Original);
        const auto actual = evaluate(in, -1, Path::Sliced, 2);
        compare(actual, reference, "d=" + std::to_string(in.physical) + " cell=" + std::to_string(in.Lx) + "x" + std::to_string(in.Ly));
        finite_difference(in, -1, actual);
    }
    const auto in = fixture(3);
    for(int kind : {1, 2}) {
        const auto result = evaluate(in, -1, Path::Sliced, 2, kind);
        check(std::abs(result.value - (kind == 1 ? 1. + .73 - .41 + 1.19 : 0.)) < 3.e-11, "identity/zero energy");
        for(const auto& site : result.gradient) for(const auto& group : site) for(double value : group)
            check(std::abs(value) < 3.e-10, "identity/zero derivative");
    }
}

void alias_case()
{
    Inputs in = fixture(2, 1, 1, true);
    in.sites[0][4] = in.sites[0][2]; in.sites[0][8] = in.sites[0][5]; in.sites[0][7] = in.sites[0][6];
    const auto independent = evaluate(in, -1, Path::Sliced, 2);
    const auto actual = evaluate(in, -1, Path::Sliced, 2, 0, true);
    compare(actual, evaluate(in, -1, Path::Original, 8, 0, true), "aliased handles");
    for(const auto pair : {std::pair{2, 4}, std::pair{5, 8}, std::pair{6, 7}}) {
        for(std::size_t i = 0; i < actual.gradient[0][pair.first].size(); ++i)
            check(std::abs(actual.gradient[0][pair.first][i] - independent.gradient[0][pair.first][i]
                           - independent.gradient[0][pair.second][i]) < 5.e-10, "aliased adjoint accumulation");
    }
}

Evaluation one_site_evaluation(bool eager)
{
    Evaluation result;
    stan::math::nested_rev_autodiff tape;
    auto env = environment<true>(fixture());
    Xped::OneSiteObservable<double, Symmetry> op(env->cell().pattern);
    op.data[0] = tensor<1, 1>({2}, {2}, {.4, .2, .2, -.7});
    if(eager) env->computeRDM<true>();
    auto loss = Xped::avg(*env, op).sum();
    check(env->RDM_COMPUTED() == eager, "one-site lazy path preserves caller RDM state");
    result.value = loss.val();
    stan::math::grad(loss.vi_);
    result.gradient.resize(1);
    result.gradient[0] = {payload(env->A->As[0].adj()), payload(env->C1s[0].adj()), payload(env->C2s[0].adj()),
        payload(env->C3s[0].adj()), payload(env->C4s[0].adj()), payload(env->T1s[0].adj()),
        payload(env->T2s[0].adj()), payload(env->T3s[0].adj()), payload(env->T4s[0].adj())};
    return result;
}

void integration_cases()
{
    compare(one_site_evaluation(false), one_site_evaluation(true), "lazy one-site value and A/C/T adjoints");
    using U1 = Xped::Sym::U1<Xped::Sym::SpinU1>;
    static_assert(!Xped::detail::supports_sliced_energy<double, U1, 2, false, double>);
    Xped::UnitCell cell(1, 1);
    Xped::Qbasis<U1, 1> aux, phys;
    aux.push_back({0}, 1); phys.push_back({0}, 2);
    Xped::TMatrix<Xped::Qbasis<U1, 1>> ab(cell.pattern), pb(cell.pattern);
    ab.setConstant(aux); pb.setConstant(phys);
    auto psi = std::make_shared<Xped::iPEPS<double, U1, false>>(cell, 1, ab, ab, pb, Xped::Opts::DiscreteSym::None);
    psi->As[0].setConstant(1.); psi->updateAdags();
    Xped::CTM<double, U1, 2, false, false> env(psi, 1, Xped::Opts::CTM_INIT::FROM_TRIVIAL);
    env.init();
    Xped::TwoSiteObservable<double, U1> op(cell.pattern, Xped::Opts::Bond::H);
    op.data_h[0] = Xped::Tensor<double, 2, 2, U1, false>::Identity({phys, phys}, {phys, phys});
    const auto& const_env = env;
    const double value = Xped::avg_sliced(const_env, op, 3)[0].sum();
    check(std::abs(value - 1.) < 1.e-12, "U1 two-site lazy fallback identity");
    check(!env.RDM_COMPUTED(), "U1 fallback preserves caller RDM state");
    env.computeRDM();
    check(std::abs(value - Xped::avg_unsliced(const_env, op)[0].sum()) < 1.e-12, "U1 fallback matches reference");

    auto invalid_env = environment<false>(fixture());
    auto invalid_op = observable(fixture(), 0);
    bool rejected = false;
    try { Xped::avg_sliced(*invalid_env, invalid_op, 0); }
    catch(const std::invalid_argument&) { rejected = true; }
    check(rejected, "zero block width is rejected");
}
} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    main_cases();
    extended_cases();
    alias_case();
    integration_cases();
    std::printf("Energy slicing: %d checks, %d failures; max value %.3e gradient %.3e FD %.3e\n",
                checks, failures, max_value_error, max_gradient_error, max_fd_error);
    return failures ? 1 : 0;
}
