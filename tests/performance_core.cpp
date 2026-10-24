// Numerical and metadata regressions for allocation-reducing core paths.
// Public APIs only: this source also runs against the pre-optimization implementation.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <type_traits>
#include <utility>

#include "stan/math/rev.hpp"
#include "Xped/Util/Macros.hpp"
#include "Xped/Util/Mpi.hpp"
#include "Xped/Interfaces/PlainInterface.hpp"
#include "Xped/Core/Qbasis.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Core/AdjointOp.hpp"
#include "Xped/Core/Tensor.hpp"
#include "Xped/AD/ADTensor.hpp"

static int failures = 0;
#define CHECK(condition) do { if(!(condition)) { std::printf("FAIL line %d: %s\n", __LINE__, #condition); ++failures; } } while(false)

// Independent scalar-loop reference: no Eigen GEMM or Xped tensor multiplication.
template <typename A, typename B>
auto reference_product(const A& a, const B& b)
{
    using Scalar = std::common_type_t<typename A::Scalar, typename B::Scalar>;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> c(a.rows(), b.cols());
    c.setZero();
    CHECK(a.cols() == b.rows());
    for(Eigen::Index j = 0; j < b.cols(); ++j)
        for(Eigen::Index k = 0; k < a.cols(); ++k)
            for(Eigen::Index i = 0; i < a.rows(); ++i) c(i, j) += a(i, k) * b(k, j);
    return c;
}

template <typename A, typename B>
double matrix_error(const A& a, const B& b)
{
    CHECK(a.rows() == b.rows() && a.cols() == b.cols());
    if(a.rows() != b.rows() || a.cols() != b.cols()) return 1.;
    return (a - b).norm() / (1. + b.norm());
}

template <typename Basis>
void check_offsets(const Basis& basis)
{
    for(const auto& q : basis.qs()) {
        const auto& trees = basis.tree(q);
        std::size_t total = 0;
        for(const auto& tree : trees) total += tree.dim;
        std::size_t left = 0;
        for(const auto& tree : trees) {
            CHECK(basis.leftOffset(tree) == left);
            CHECK(basis.rightOffset(tree) == total - left - tree.dim);
            CHECK(basis.leftOffset(tree, tree.dims) == left);
            CHECK(basis.rightOffset(tree, tree.dims) == total - left - tree.dim);
#ifndef XPED_PEDANTIC_ASSERTS
            // Non-pedantic equality ignores duality, so cache hashing must also do so.
            auto equivalent = tree;
            equivalent.IS_DUAL[0] = !equivalent.IS_DUAL[0];
            CHECK(basis.leftOffset(equivalent) == left);
            CHECK(basis.rightOffset(equivalent) == total - left - tree.dim);
#endif
            left += tree.dim;
        }
    }
}

template <typename T>
void check_storage(const T& tensor)
{
    std::size_t offset = 0;
    for(std::size_t i = 0; i < tensor.sector().size(); ++i) {
        const auto q = tensor.sector(i);
        const auto block = tensor.block(i);
        CHECK(block.rows() == static_cast<Eigen::Index>(tensor.coupledDomain().inner_dim(q)));
        CHECK(block.cols() == static_cast<Eigen::Index>(tensor.coupledCodomain().inner_dim(q)));
        CHECK(block.data() == tensor.data() + offset);
        CHECK(tensor.dict().at(q) == i);
        offset += block.size();
    }
    CHECK(offset == tensor.plainSize());
    check_offsets(tensor.coupledDomain());
    check_offsets(tensor.coupledCodomain());
}

template <typename Basis>
void check_basis_lifecycle(const Basis& first, const Basis& second)
{
    auto basis = first;
    check_offsets(basis);
    basis.push_back(first.qn(0), 2);
    check_offsets(basis);
    basis.sort();
    check_offsets(basis);
    check_offsets(basis.conj());
    auto truncated = basis;
    truncated.truncate(truncated.unordered_qs(), truncated.Nq() + 1);
    check_offsets(truncated);

    auto fused = basis.combine(second).combine(first);
    check_offsets(fused);
    auto copied = fused;
    check_offsets(copied);
    auto moved = std::move(copied);
    check_offsets(moved);
    copied = fused;
    moved = std::move(copied);
    check_offsets(moved);

    constexpr auto flags = yas::mem | yas::binary;
    const auto buffer = yas::save<flags>(fused);
    auto restored = second.combine(first).combine(second);
    check_offsets(restored);
    // YAS map loading requires an empty destination; it does not replace existing keys.
    restored.clear();
    yas::load<flags>(buffer, restored);
    CHECK(restored == fused);
    check_offsets(restored);

    // This mutable reference can survive arbitrarily many intervening offset reads.
    auto& escaped = moved.allTrees();
    check_offsets(moved);
    moved = fused; // Assignment must preserve correctness of the retained map reference.
    check_offsets(moved);
    for(auto& [q, trees] : escaped) std::reverse(trees.begin(), trees.end());
    check_offsets(moved);
    for(auto& [q, trees] : escaped) {
        std::reverse(trees.begin(), trees.end());
        if(trees.size() > 1) {
            ++trees.front().dim;
            ++trees.front().dims[0];
        }
    }
    // External mutation need not update sector dimensions; only query tree offsets.
    check_offsets(moved);
    basis.clear();
    basis.push_back(second.qn(0), 3);
    check_offsets(basis);
    CHECK(basis.Nq() == 1);
}

template <typename Left, typename Right>
void check_product(const Left& left, const Right& right)
{
    const auto result = left * right;
    check_storage(result);
    using Scalar = typename decltype(result)::Scalar;
    const auto& left_dict = left.dict();
    const auto& right_dict = right.dict();
    for(std::size_t i = 0; i < result.sector().size(); ++i) {
        const auto q = result.sector(i);
        const auto li = left_dict.find(q), ri = right_dict.find(q);
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> expected(result.block(i).rows(), result.block(i).cols());
        expected.setZero();
        if(li != left_dict.end() && ri != right_dict.end())
            expected = reference_product(left.block(li->second), right.block(ri->second));
        CHECK(matrix_error(result.block(i), expected) < 2.e-12);
    }
}

template <typename Scalar, typename Symmetry>
void check_forward(const Xped::Qbasis<Symmetry, 1>& b1, const Xped::Qbasis<Symmetry, 1>& b2)
{
    using Left = Xped::Tensor<Scalar, 2, 1, Symmetry, false>;
    using Right = Xped::Tensor<Scalar, 1, 2, Symmetry, false>;
    const auto middle = b1.combine(b2).forgetHistory();
    Left a({{b1, b2}}, {{middle}});
    Right b({{middle}}, {{b2, b1}});
    std::mt19937 rng(1729);
    a.setRandom(rng);
    b.setRandom(rng);
    check_product(a, b);
    check_product(a.adjoint(), a);
    check_product(a, a.adjoint());
    check_product(b.adjoint(), a.adjoint());
    check_product(Scalar(1.25) * a, Scalar(-0.75) * b);
    check_product((a + a), b);

    check_storage(a);
    Left copied(a);
    check_storage(copied);
    Left moved(std::move(copied));
    check_storage(moved);
    copied = a;
    moved = std::move(copied);
    check_storage(moved);
    Left initialized(a.uncoupledDomain(), a.uncoupledCodomain(), a.data(), a.plainSize(), a.world());
    check_storage(initialized);
    for(std::size_t i = 0; i < a.sector().size(); ++i) CHECK(matrix_error(initialized.block(i), a.block(i)) == 0.);

    constexpr auto flags = yas::mem | yas::binary;
    const auto buffer = yas::save<flags>(a);
    Left restored;
    yas::load<flags>(buffer, restored);
    check_storage(restored);
    for(std::size_t i = 0; i < a.sector().size(); ++i) CHECK(matrix_error(restored.block(i), a.block(i)) == 0.);
    restored.setZero();
    restored.setZero(); // Repeated resize cannot duplicate metadata.
    check_storage(restored);
    restored.set_data(a.data(), a.plainSize());
    check_storage(restored);
    restored.storage().clear();
    for(std::size_t i = 0; i < a.sector().size(); ++i) restored.storage().push_back(a.sector(i), a.block(i));
    check_storage(restored);
    for(std::size_t i = 0; i < a.sector().size(); ++i) CHECK(matrix_error(restored.block(i), a.block(i)) == 0.);

    const auto product = a * b;
    for(std::size_t i = 0; i < product.sector().size(); ++i) {
        const auto q = product.sector(i);
        Eigen::Index row = 0;
        for(const auto& f1 : product.coupledDomain().tree(q)) {
            Eigen::Index col = 0;
            for(const auto& f2 : product.coupledCodomain().tree(q)) {
                CHECK(matrix_error(product.subMatrix(f1, f2), product.block(i).block(row, col, f1.dim, f2.dim)) == 0.);
                col += f2.dim;
            }
            row += f1.dim;
        }
    }
    const auto permuted = product.template permute<0, 0, 2, 1, 3>();
    const auto back = permuted.template permute<0, 0, 2, 1, 3>();
    for(std::size_t i = 0; i < product.sector().size(); ++i)
        CHECK(matrix_error(back.block(back.dict().at(product.sector(i))), product.block(i)) < 2.e-11);

    // Outer spaces admit more sectors than the middle: unmatched output stays zero.
    Xped::Qbasis<Symmetry, 1> narrow;
    narrow.push_back(b1.qn(0), 2);
    using Matrix = Xped::Tensor<Scalar, 1, 1, Symmetry, false>;
    Matrix sparse_left({{b1}}, {{narrow}}), sparse_right({{narrow}}, {{b1}});
    sparse_left.setRandom(rng);
    sparse_right.setRandom(rng);
    check_product(sparse_left, sparse_right);
    const auto sparse_result = sparse_left * sparse_right;
    CHECK(sparse_result.sector().size() == b1.Nq());
    for(std::size_t i = 0; i < sparse_result.sector().size(); ++i)
        if(sparse_result.sector(i) != narrow.qn(0)) CHECK(sparse_result.block(i).squaredNorm() == 0.);
}

template <typename Scalar, typename Symmetry>
void check_ad(const Xped::Qbasis<Symmetry, 1>& basis)
{
    using Plain = Xped::Tensor<Scalar, 1, 1, Symmetry, false>;
    using Active = Xped::Tensor<Scalar, 1, 1, Symmetry, true>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Plain a0({{basis}}, {{basis}}), b0({{basis}}, {{basis}});
    std::mt19937 rng(2027);
    a0.setRandom(rng);
    b0.setRandom(rng);
    // Both active, left active, right active, one active object used on both sides.
    for(int mode = 0; mode < 4; ++mode) {
        stan::math::nested_rev_autodiff nested;
        Active a(a0), b(b0);
        auto c = [&]() {
            if(mode == 1) return a * b0;
            if(mode == 2) return a0 * b;
            if(mode == 3) return a * a;
            return a * b;
        }();
        auto loss = (c * c.adjoint()).trace();
        stan::math::grad(loss.vi_);
        double expected_loss = 0.;
        for(std::size_t i = 0; i < a0.sector().size(); ++i) {
            const Matrix aa = a0.block(i);
            const Matrix bb = mode == 3 ? aa : Matrix(b0.block(i));
            const auto cc = reference_product(aa, bb);
            const double weight = Symmetry::degeneracy(a0.sector(i));
            expected_loss += weight * cc.squaredNorm();
            Matrix ga = 2. * weight * reference_product(cc, bb.adjoint());
            Matrix gb = 2. * weight * reference_product(aa.adjoint(), cc);
            if(mode == 3) { ga += gb; gb.setZero(); }
            if(mode == 1) gb.setZero();
            if(mode == 2) ga.setZero();
            CHECK(matrix_error(a.adj().block(i), ga) < 3.e-12);
            CHECK(matrix_error(b.adj().block(i), gb) < 3.e-12);
        }
        CHECK(std::abs(loss.val() - expected_loss) < 2.e-12 * (1. + expected_loss));
    }
}

// Supported mixed-scalar pullbacks: only the complex operand is active.
template <typename Symmetry>
void check_mixed_scalar_ad(const Xped::Qbasis<Symmetry, 1>& basis)
{
    using Real = Xped::Tensor<double, 1, 1, Symmetry, false>;
    using Complex = Xped::Tensor<std::complex<double>, 1, 1, Symmetry, false>;
    using Active = Xped::Tensor<std::complex<double>, 1, 1, Symmetry, true>;
    Real passive({{basis}}, {{basis}});
    Complex z0({{basis}}, {{basis}});
    std::mt19937 rng(54321);
    passive.setRandom(rng);
    z0.setRandom(rng);
    for(int mode = 0; mode < 2; ++mode) {
        stan::math::nested_rev_autodiff nested;
        Active z(z0);
        auto c = mode == 0 ? passive * z : z * passive;
        auto loss = (c * c.adjoint()).trace();
        stan::math::grad(loss.vi_);
        double expected_loss = 0.;
        for(std::size_t i = 0; i < z0.sector().size(); ++i) {
            const Eigen::MatrixXd a = passive.block(i);
            const Eigen::MatrixXcd zz = z0.block(i);
            const auto cc = mode == 0 ? reference_product(a, zz) : reference_product(zz, a);
            const double weight = Symmetry::degeneracy(z0.sector(i));
            expected_loss += weight * cc.squaredNorm();
            Eigen::MatrixXcd gradient;
            if(mode == 0) gradient = 2. * weight * reference_product(a.adjoint(), cc);
            else gradient = 2. * weight * reference_product(cc, a.adjoint());
            CHECK(matrix_error(z.adj().block(i), gradient) < 3.e-12);
        }
        CHECK(std::abs(loss.val() - expected_loss) < 2.e-12 * (1. + expected_loss));
    }
}

template <typename Symmetry>
void run_symmetry(const char* name, const Xped::Qbasis<Symmetry, 1>& b1, const Xped::Qbasis<Symmetry, 1>& b2)
{
    const int before = failures;
    check_basis_lifecycle(b1, b2);
    check_forward<double>(b1, b2);
    check_forward<std::complex<double>>(b1, b2);
    check_ad<double>(b1);
    check_ad<std::complex<double>>(b1);
    check_mixed_scalar_ad(b1);
    Xped::Tensor<double, 1, 1, Symmetry, false> real({{b1}}, {{b1}});
    Xped::Tensor<std::complex<double>, 1, 1, Symmetry, false> complex({{b1}}, {{b1}});
    std::mt19937 rng(23);
    real.setRandom(rng);
    complex.setRandom(rng);
    check_product(real, complex);
    check_product(complex, real);
    std::printf("%s core regression: %d failures\n", name, failures - before);
}

int main()
{
    using U1 = Xped::Sym::U1<Xped::Sym::SpinU1>;
    using SU2 = Xped::Sym::SU2<Xped::Sym::SpinSU2>;
    Xped::Qbasis<U1, 1> u1a, u1b;
    u1a.push_back({-1}, 3); u1a.push_back({0}, 2); u1a.push_back({1}, 1);
    u1b.push_back({-1}, 1); u1b.push_back({1}, 2);
    Xped::Qbasis<SU2, 1> su2a, su2b;
    su2a.push_back({1}, 3); su2a.push_back({2}, 2); su2a.push_back({3}, 1);
    su2b.push_back({1}, 1); su2b.push_back({2}, 2);
    run_symmetry("U1", u1a, u1b);
    run_symmetry("SU2", su2a, su2b);
    std::printf("Performance core: %d failures\n", failures);
    return failures ? 1 : 0;
}
