#include <limits>

#include "Xped/AD/finite_diff.hpp"
#include "Xped/PEPS/CTMSolver.hpp"
#include "Xped/PEPS/LinearAlgebra.hpp"
#include "Xped/Util/Logging.hpp"
#include "Xped/Util/Stopwatch.hpp"

namespace Xped {

template <typename Scalar, typename Symmetry, typename HamScalar, bool ALL_OUT_LEGS, Opts::CTMCheckpoint CPOpts, std::size_t TRank>
template <bool AD>
typename ScalarTraits<Scalar>::Real
CTMSolver<Scalar, Symmetry, HamScalar, ALL_OUT_LEGS, CPOpts, TRank>::solve(std::shared_ptr<iPEPS<Scalar, Symmetry, ALL_OUT_LEGS>> Psi,
                                                                           Scalar* gradient,
                                                                           Hamiltonian<HamScalar, Symmetry>& H)
{
    util::Stopwatch<> total_t;
    Jack.set_A(Psi);
    if(not Psi->checkSym()) { Log::warning("CTMSolver: iPEPS is not C4v symmetric"); }

    if constexpr(AD) {
        Log::on_entry(opts.verbosity,
                      "  CTMSolver(χ={}, {}): UnitCell=({}x{}), init={}, reinit env={}, max steps(untracked)={}, steps(tracked)={}",
                      opts.chi,
                      Symmetry::name(),
                      Jack.cell().Lx,
                      Jack.cell().Ly,
                      fmt::streamed(Jack.init_mode()),
                      REINIT_ENV,
                      opts.max_presteps,
                      opts.track_steps);
    } else {
        Log::on_entry(opts.verbosity,
                      "  CTMSolver(χ={}, {}): UnitCell=({}x{}), init={}, reinit env={}, max steps={}",
                      opts.chi,
                      Symmetry::name(),
                      Jack.cell().Lx,
                      Jack.cell().Ly,
                      fmt::streamed(Jack.init_mode()),
                      REINIT_ENV,
                      opts.max_presteps);
    }
    Log::on_entry(opts.verbosity, "   Environment for {}", Psi->info());
    if(REINIT_ENV) {
        Jack = CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, false>(Psi, opts.chi, opts.init);
        Jack.init();
    } else {
        Jack.updateChi(opts.chi);
    }

    double E = std::numeric_limits<typename ScalarTraits<Scalar>::Real>::quiet_NaN();
    double Eprev = 1000.;
    double Eprevprev = 1000.;
    util::Stopwatch<> pre_t;
    std::size_t used_steps = 0ul;
    for(std::size_t step = 0; step < opts.max_presteps; ++step) {
        util::Stopwatch<> move_t;
        Jack.grow_all();
        if constexpr(detail::supports_sliced_energy<Scalar, Symmetry, TRank, ALL_OUT_LEGS, HamScalar>) {
            Jack.HAS_RDM = false;
        } else {
            Jack.computeRDM();
        }
        auto Hobs = H.asObservable();
        auto [E_h, E_v, E_d1, E_d2] = avg(Jack, Hobs);
        E = (E_h.sum() + E_v.sum() + E_d1.sum() + E_d2.sum()) / Jack.cell().uniqueSize();
        if constexpr(ALL_OUT_LEGS) { E = 2. * E; }
        step == 0 ? Log::per_iteration(opts.verbosity, "  {: >3} {:2d}: E={:2.8f}, t={}", "▷", step, E, move_t.time_string())
                  : Log::per_iteration(
                        opts.verbosity, "  {: >3} {:2d}: E={:2.8f}, conv={:2.10g}, t={}", "▷", step, E, std::abs(E - Eprev), move_t.time_string());
        if(std::abs(E - Eprev) < opts.tol_E) {
            used_steps = step;
            break;
        }
        if(std::abs(E - Eprevprev) < opts.tol_E) {
            Log::per_iteration(opts.verbosity, "  {: >3} Oscillation -> break", "•");
            used_steps = step;
            break;
        }
        Eprevprev = Eprev;
        Eprev = E;
    }
    Jack.checkHermiticity();
    if(not Jack.checkSym()) { Log::warning("CTMSolver: CTM environment is not C4v symmetric"); }
    if(used_steps == 0) { used_steps = opts.max_presteps; }

    auto pre_time = pre_t.time_string();
    Log::per_iteration(opts.verbosity, "  {: >3} pre steps: {}", "•", pre_time);
    if constexpr(not AD) {
        REINIT_ENV = true;
        Log::on_exit(
            opts.verbosity, "  CTMSolver(χ={}({}), runtime={}[{} steps]): E={:.8f}", opts.chi, Jack.fullChi(), total_t.time_string(), used_steps, E);
        return E;
    } else {
        auto do_tracked_steps = [this, &H](auto Psi_ad) {
            constexpr bool ENABLE_AD = std::decay_t<decltype(Psi_ad)>::ENABLE_AD;
            CTM<Scalar, Symmetry, TRank, ALL_OUT_LEGS, ENABLE_AD, CPOpts> Jim(Jack);
            Jim.set_A(Psi_ad);
            Jim.Psi()->updateAdags();
            if constexpr(detail::supports_sliced_energy<Scalar, Symmetry, TRank, ALL_OUT_LEGS, HamScalar>) {
                for(std::size_t step = 0; step < opts.track_steps; ++step) { Jim.template grow_all<ENABLE_AD>(); }
                Jim.HAS_RDM = false;
            } else {
                Jim.solve(opts.track_steps);
            }
            auto Hobs = H.asObservable();
            auto [E_h, E_v, E_d1, E_d2] = avg(Jim, Hobs);
            auto res = (E_h.sum() + E_v.sum() + E_d1.sum() + E_d2.sum()) / Jim.cell().uniqueSize();
            return res;
        };

        iPEPS<Scalar, Symmetry, ALL_OUT_LEGS, true> Psi_ad = *Psi;

        stan::math::nested_rev_autodiff nested;
        util::Stopwatch<> forward_t;
        auto res = do_tracked_steps(Psi_ad);
        auto forward_time = forward_t.time_string();
        Log::per_iteration(opts.verbosity, "  {: >3} forward pass: {}", "•", forward_time);
        util::Stopwatch<> backward_t;
        stan::math::grad(res.vi_);
        auto backward_time = backward_t.time_string();
        auto grad = Psi_ad.graddata();
        // auto grad_ad = *Psi;
        // grad_ad.set_data(grad.data());
        // if(not grad_ad.checkSym()) { Log::warning("CTMSolver: gradient of iPEPS is not C4v symmetric"); }
        // grad_ad.debug_info();
        Log::per_iteration(opts.verbosity, "  {: >3} backward pass: {}", "•", backward_time);

        if(opts.COMPARE_TO_FD) {
            util::Stopwatch<> fd_t;
            auto grad_A_fd = internal::finite_diff_gradient(do_tracked_steps, *Psi);
            auto fd_time = fd_t.time_string();
            auto grad_fd = grad_A_fd.data();
            auto diff = std::transform_reduce(
                grad_fd.begin(),
                grad_fd.end(),
                grad.begin(),
                0.,
                [](auto sum, auto d) { return sum + d; },
                [](auto d1, auto d2) { return std::abs(d1 - d2); });
            Log::per_iteration(opts.verbosity, "  {: >3} finite-difference gradient check: {} ({})", "•", diff, fd_time);
            Log::debug(opts.verbosity, "  {: >3} fd gradient:{}", "•", grad_fd);
            Log::debug(opts.verbosity, "  {: >3} ad gradient:{}", "•", grad);
        }

        for(std::size_t i = 0; i < grad.size(); ++i) { gradient[i] = grad[i]; }
        if constexpr(ScalarTraits<Scalar>::IS_COMPLEX()) {
            grad_norm = std::sqrt(
                std::inner_product(gradient,
                                   gradient + Psi->plainSize(),
                                   gradient,
                                   0.,
                                   std::plus<>(),
                                   [](const std::complex<double>& c1, const std::complex<double>& c2) { return std::real(c1 * std::conj(c2)); }));
        } else {
            grad_norm = std::sqrt(std::inner_product(gradient, gradient + Psi->plainSize(), gradient, 0.));
        }
        REINIT_ENV = grad_norm < opts.reinit_env_tol ? false : true;
        Log::on_exit(opts.verbosity,
                     "  CTMSolver(χ={}({}), runtime={} [{} steps, pre={}, forward={}, backward={}]): E={:.8f}, |∇|={:.1e}",
                     opts.chi,
                     Jack.fullChi(),
                     total_t.time_string(),
                     used_steps,
                     pre_time,
                     forward_time,
                     backward_time,
                     E,
                     grad_norm);
        return E;
    }
}

} // namespace Xped

#if __has_include("CTMSolver.gen.cpp")
#    include "CTMSolver.gen.cpp"
#endif
