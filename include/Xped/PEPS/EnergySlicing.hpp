#ifndef XPED_ENERGY_SLICING_HPP_
#define XPED_ENERGY_SLICING_HPP_

// Exact real-U0 observable contractions. Physical legs close before boundary
// enlargement; operator channels and free-chi blocks replay on the native tape.
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "Xped/AD/reverse_pass_callback_alloc.hpp"
#include "Xped/PEPS/CornerBlocking.hpp"
#include "Xped/PEPS/Bonds.hpp"

#ifdef XPED_USE_EIGEN_MATRIX_LIB
namespace Xped::detail::energy_slicing {
using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;

// Matrix handles live on Xped's native tape. Allocating callbacks release their
// captures when a nested replay finishes; no separate autodiff engine is used.
struct MatrixHandle {
    struct Node { Matrix value, gradient; };
    std::shared_ptr<Node> node;
    explicit MatrixHandle(Matrix value) : node(std::make_shared<Node>(Node{std::move(value), {}})) {}
    MatrixHandle() = default;
    const Matrix& val() const { return node->value; }
    Matrix& adj() const {
        if(node->gradient.size() == 0) node->gradient = Matrix::Zero(val().rows(), val().cols());
        return node->gradient;
    }
};
using Edge = std::vector<MatrixHandle>;

template <bool Track, typename F>
auto checkpoint(F function)
{
    if constexpr(!Track) return function.template operator()<false>();
    else {
        stan::math::var result(function.template operator()<false>());
        Xped::reverse_pass_callback_alloc([function, result]() mutable {
            if(result.adj() == 0.) return;
            stan::math::nested_rev_autodiff nested;
            auto replay = function.template operator()<true>();
            replay.adj() += result.adj();
            stan::math::grad();
        });
        return result;
    }
}

template <bool Track, typename Tensor>
MatrixHandle adapt(const Tensor& tensor, Index rows, Index cols, std::vector<Index> offsets)
{
    const auto block = corner_value(tensor).block(0);
    Matrix value(rows, cols);
    for(Index i = 0; i < value.size(); ++i) value.data()[i] = block.data()[offsets[i]];
    MatrixHandle result(std::move(value));
    if constexpr(Track) {
        Xped::reverse_pass_callback_alloc([tensor, result, offsets = std::move(offsets)]() mutable {
            decltype(auto) gradient = tensor.adj().block(0);
            for(Index i = 0; i < result.val().size(); ++i) gradient.data()[offsets[i]] += result.adj().data()[i];
        });
    }
    return result;
}

template <std::size_t Leg, typename Tensor>
Index dimension(const Tensor& tensor) { return static_cast<Index>(corner_leg_basis<Leg>(tensor).dim()); }

template <bool Track, typename Tensor>
MatrixHandle corner(const Tensor& tensor, bool transpose = false)
{
    const Index a = dimension<0>(tensor), b = dimension<1>(tensor);
    const Index rows = transpose ? b : a, cols = transpose ? a : b;
    std::vector<Index> offsets(rows * cols);
    for(Index j = 0; j < cols; ++j) for(Index i = 0; i < rows; ++i)
        offsets[i + rows * j] = transpose ? j + a * i : i + a * j;
    return adapt<Track>(tensor, rows, cols, std::move(offsets));
}

template <bool Track, int Direction, typename Tensor>
Edge edge(const Tensor& tensor)
{
    Index rows, cols, ket, bra;
    if constexpr(Direction == 0 || Direction == 3) {
        rows = dimension<Direction == 0 ? 0 : 1>(tensor);
        cols = dimension<Direction == 0 ? 1 : 0>(tensor);
        ket = dimension<2>(tensor); bra = dimension<3>(tensor);
    } else {
        rows = dimension<Direction == 1 ? 2 : 3>(tensor);
        cols = dimension<Direction == 1 ? 3 : 2>(tensor);
        ket = dimension<0>(tensor); bra = dimension<1>(tensor);
    }
    Edge result;
    for(Index k = 0; k < ket; ++k) for(Index b = 0; b < bra; ++b) {
        std::vector<Index> offsets(rows * cols);
        for(Index j = 0; j < cols; ++j) for(Index i = 0; i < rows; ++i) {
            Index offset;
            if constexpr(Direction == 0) offset = i + rows * (j + cols * (k + ket * b));
            else if constexpr(Direction == 1) offset = k + ket * (b + bra * (i + rows * j));
            else if constexpr(Direction == 2) offset = k + ket * (b + bra * (j + cols * i));
            else offset = j + cols * (i + rows * (k + ket * b));
            offsets[i + rows * j] = offset;
        }
        result.push_back(adapt<Track>(tensor, rows, cols, std::move(offsets)));
    }
    return result;
}

template <typename Symmetry, bool AD>
struct Patch {
    using Ket = Tensor<double, 2, 3, Symmetry, AD>;
    using Bra = Tensor<double, 3, 2, Symmetry, AD>;
    std::vector<Ket> ket;
    std::vector<Bra> bra;
    Tensor<double, 0, 2, Symmetry, AD> c0;
    Tensor<double, 1, 1, Symmetry, AD> c1, c3;
    Tensor<double, 2, 0, Symmetry, AD> c2;
    std::vector<Tensor<double, 1, 3, Symmetry, AD>> top, left;
    std::vector<Tensor<double, 3, 1, Symmetry, AD>> right, bottom;
    int width, height;
};

template <bool Track, typename Ket, typename Bra>
MatrixHandle bulk(const Ket& ket, const Bra& bra, const Matrix& op, int rotation)
{
    const std::array<Index, 4> kd{dimension<0>(ket), dimension<1>(ket), dimension<2>(ket), dimension<3>(ket)};
    const std::array<Index, 4> bd{dimension<0>(bra), dimension<1>(bra), dimension<3>(bra), dimension<4>(bra)};
    const Index pk = dimension<4>(ket), pb = dimension<2>(bra);
    if(op.rows() != pk || op.cols() != pb) throw std::invalid_argument("Energy operator physical dimensions disagree");
    std::array<Index, 4> q;
    for(int k = 0; k < 4; ++k) q[k] = kd[k] * bd[k];
    const auto ki = [kd](const std::array<Index, 4>& v, Index p) {
        return v[0] + kd[0] * (v[1] + kd[1] * (v[2] + kd[2] * (v[3] + kd[3] * p)));
    };
    const auto bi = [bd, pb](const std::array<Index, 4>& v, Index p) {
        return v[0] + bd[0] * (v[1] + bd[1] * (p + pb * (v[2] + bd[2] * v[3])));
    };
    auto visit = [q, bd, rotation](auto function) {
        for(Index l = 0; l < q[0]; ++l) for(Index u = 0; u < q[1]; ++u)
            for(Index r = 0; r < q[2]; ++r) for(Index d = 0; d < q[3]; ++d) {
                const std::array<Index, 4> v{l, u, r, d};
                std::array<Index, 4> k, b;
                for(int j = 0; j < 4; ++j) { k[j] = v[j] / bd[j]; b[j] = v[j] % bd[j]; }
                function(k, b, v[rotation] * q[(rotation + 1) % 4] + v[(rotation + 1) % 4],
                         v[(rotation + 2) % 4] * q[(rotation + 3) % 4] + v[(rotation + 3) % 4]);
            }
    };
    Matrix value = Matrix::Zero(q[rotation] * q[(rotation + 1) % 4], q[(rotation + 2) % 4] * q[(rotation + 3) % 4]);
    const auto kval = corner_value(ket).block(0), bval = corner_value(bra).block(0);
    visit([&](const auto& k, const auto& b, Index row, Index col) {
        for(Index p = 0; p < pk; ++p) for(Index s = 0; s < pb; ++s)
            value(row, col) += kval.data()[ki(k, p)] * op(p, s) * bval.data()[bi(b, s)];
    });
    MatrixHandle result(std::move(value));
    if constexpr(Track) {
        Xped::reverse_pass_callback_alloc([ket, bra, op, pk, pb, ki, bi, visit, result]() mutable {
            const auto kv = ket.val().block(0), bv = bra.val().block(0);
            decltype(auto) kg = ket.adj().block(0);
            decltype(auto) bg = bra.adj().block(0);
            visit([&](const auto& k, const auto& b, Index row, Index col) {
                const double gradient = result.adj()(row, col);
                for(Index p = 0; p < pk; ++p) for(Index s = 0; s < pb; ++s) {
                    const double coefficient = gradient * op(p, s);
                    kg.data()[ki(k, p)] += coefficient * bv.data()[bi(b, s)];
                    bg.data()[bi(b, s)] += coefficient * kv.data()[ki(k, p)];
                }
            });
        });
    }
    return result;
}

inline Matrix joined(const MatrixHandle& c, const Edge& previous, const Edge& next, Index begin, Index width)
{
    const Index n = next[0].val().cols(), qu = next.size();
    Matrix result(width * n, previous.size() * qu);
    for(Index l = 0; l < static_cast<Index>(previous.size()); ++l) {
        Matrix left = previous[l].val().middleRows(begin, width) * c.val();
        for(Index u = 0; u < qu; ++u) {
            Matrix z = left * next[u].val();
            for(Index i = 0; i < width; ++i) for(Index j = 0; j < n; ++j) result(i * n + j, l * qu + u) = z(i, j);
        }
    }
    return result;
}

inline Matrix corner_piece(const MatrixHandle& c, const Edge& previous, const Edge& next, const MatrixHandle& a,
                           Index qr, Index qd, Index begin, Index width)
{
    const Index n = next[0].val().cols();
    Matrix z = joined(c, previous, next, begin, width) * a.val();
    Matrix result(width * qd, n * qr);
    for(Index i = 0; i < width; ++i) for(Index j = 0; j < n; ++j)
        for(Index r = 0; r < qr; ++r) for(Index d = 0; d < qd; ++d) result(i * qd + d, j * qr + r) = z(i * n + j, r * qd + d);
    return result;
}

inline void reverse_piece(const MatrixHandle& c, const Edge& previous, const Edge& next, const MatrixHandle& a,
                          Index qr, Index qd, Index begin, Index width, const Matrix& gradient)
{
    const Index n = next[0].val().cols(), qu = next.size();
    Matrix dz(width * n, qr * qd);
    for(Index i = 0; i < width; ++i) for(Index j = 0; j < n; ++j)
        for(Index r = 0; r < qr; ++r) for(Index d = 0; d < qd; ++d) dz(i * n + j, r * qd + d) = gradient(i * qd + d, j * qr + r);
    Matrix j = joined(c, previous, next, begin, width);
    a.adj().noalias() += j.transpose() * dz;
    Matrix dj = dz * a.val().transpose();
    for(Index l = 0; l < static_cast<Index>(previous.size()); ++l) {
        Matrix left = previous[l].val().middleRows(begin, width) * c.val(), dl = Matrix::Zero(width, c.val().cols());
        for(Index u = 0; u < qu; ++u) {
            Matrix ds(width, n);
            for(Index i = 0; i < width; ++i) for(Index k = 0; k < n; ++k) ds(i, k) = dj(i * n + k, l * qu + u);
            dl.noalias() += ds * next[u].val().transpose();
            next[u].adj().noalias() += left.transpose() * ds;
        }
        previous[l].adj().middleRows(begin, width).noalias() += dl * c.val().transpose();
        c.adj().noalias() += previous[l].val().middleRows(begin, width).transpose() * dl;
    }
}

template <bool Track>
MatrixHandle enlarged(const MatrixHandle& c, const Edge& previous, const Edge& next, const MatrixHandle& a,
                      Index qr, Index qd, Index block_size)
{
    const Index m = previous[0].val().rows();
    Matrix value(m * qd, next[0].val().cols() * qr);
    for(Index begin = 0; begin < m; begin += block_size) {
        const Index width = std::min(block_size, m - begin);
        value.middleRows(begin * qd, width * qd) = corner_piece(c, previous, next, a, qr, qd, begin, width);
    }
    MatrixHandle result(std::move(value));
    if constexpr(Track) Xped::reverse_pass_callback_alloc([c, previous, next, a, qr, qd, block_size, m, result]() {
        for(Index begin = 0; begin < m; begin += block_size) {
            const Index width = std::min(block_size, m - begin);
            reverse_piece(c, previous, next, a, qr, qd, begin, width, result.adj().middleRows(begin * qd, width * qd));
        }
    });
    return result;
}

inline Matrix closing_piece(const MatrixHandle& c, const Edge& edge, Index begin, Index width)
{
    const Index q = edge.size();
    Matrix result(edge[0].val().rows(), width * q);
    for(Index d = 0; d < q; ++d) {
        Matrix product = edge[d].val() * c.val().middleCols(begin, width);
        for(Index i = 0; i < product.rows(); ++i) for(Index j = 0; j < width; ++j) result(i, j * q + d) = product(i, j);
    }
    return result;
}

// Close the third boundary edge immediately, retaining only chi^2 D^2
// half-site outputs rather than a full enlarged corner for nearest neighbours.
template <bool Track>
MatrixHandle half_site(const MatrixHandle& c, const Edge& previous, const Edge& next, const MatrixHandle& closing_corner,
                       const Edge& closing_edge, const MatrixHandle& a, Index qr, Index block_size)
{
    const Index m = previous[0].val().rows(), qd = closing_edge.size();
    Matrix value = Matrix::Zero(closing_edge[0].val().rows(), next[0].val().cols() * qr);
    for(Index begin = 0; begin < m; begin += block_size) {
        const Index width = std::min(block_size, m - begin);
        value.noalias() += closing_piece(closing_corner, closing_edge, begin, width) * corner_piece(c, previous, next, a, qr, qd, begin, width);
    }
    MatrixHandle result(std::move(value));
    if constexpr(Track) Xped::reverse_pass_callback_alloc([c, previous, next, closing_corner, closing_edge, a, qr, qd, block_size, m, result]() {
        for(Index begin = 0; begin < m; begin += block_size) {
            const Index width = std::min(block_size, m - begin);
            Matrix n = closing_piece(closing_corner, closing_edge, begin, width);
            Matrix q = corner_piece(c, previous, next, a, qr, qd, begin, width);
            Matrix dq = n.transpose() * result.adj(), dn = result.adj() * q.transpose();
            reverse_piece(c, previous, next, a, qr, qd, begin, width, dq);
            for(Index d = 0; d < qd; ++d) {
                Matrix piece(dn.rows(), width);
                for(Index i = 0; i < dn.rows(); ++i) for(Index j = 0; j < width; ++j) piece(i, j) = dn(i, j * qd + d);
                closing_edge[d].adj().noalias() += piece * closing_corner.val().middleCols(begin, width).transpose();
                closing_corner.adj().middleCols(begin, width).noalias() += closing_edge[d].val().transpose() * piece;
            }
        }
    });
    return result;
}

template <bool Track>
auto trace_pair(const MatrixHandle& a, const MatrixHandle& b, Index q)
{
    double value = 0.;
    for(Index r = 0; r < q; ++r) for(Index i = 0; i < a.val().rows(); ++i) for(Index j = 0; j < b.val().rows(); ++j)
        value += a.val()(i, j * q + r) * b.val()(j, i * q + r);
    if constexpr(!Track) return value;
    else {
        stan::math::var result(value);
        Xped::reverse_pass_callback_alloc([a, b, q, result]() {
            for(Index r = 0; r < q; ++r) for(Index i = 0; i < a.val().rows(); ++i) for(Index j = 0; j < b.val().rows(); ++j) {
                a.adj()(i, j * q + r) += result.adj() * b.val()(j, i * q + r);
                b.adj()(j, i * q + r) += result.adj() * a.val()(i, j * q + r);
            }
        });
        return result;
    }
}

template <bool Track>
auto trace_ring(const std::array<MatrixHandle, 4>& ring)
{
    const auto& a = ring[0]; const auto& b = ring[1]; const auto& c = ring[2]; const auto& d = ring[3];
    Matrix left = a.val() * b.val(), right = c.val() * d.val();
    const double value = (left.array() * right.transpose().array()).sum();
    if constexpr(!Track) return value;
    else {
        stan::math::var result(value);
        Xped::reverse_pass_callback_alloc([a, b, c, d, result]() {
            Matrix left = a.val() * b.val(), right = c.val() * d.val();
            a.adj().noalias() += result.adj() * (b.val() * right).transpose();
            b.adj().noalias() += result.adj() * (right * a.val()).transpose();
            c.adj().noalias() += result.adj() * (d.val() * left).transpose();
            d.adj().noalias() += result.adj() * (left * c.val()).transpose();
        });
        return result;
    }
}

struct Product { Matrix left, right; };
template <typename Operator>
std::vector<Product> products(const Operator& op)
{
    const Index p = dimension<0>(op), q = dimension<1>(op);
    if(dimension<2>(op) != p || dimension<3>(op) != q) throw std::invalid_argument("Energy operator must preserve physical dimensions");
    const auto block = op.block(0);
    Matrix reshaped(p * p, q * q);
    for(Index s = 0; s < p; ++s) for(Index u = 0; u < q; ++u) for(Index k = 0; k < p; ++k) for(Index t = 0; t < q; ++t)
        reshaped(k + p * s, t + q * u) = block.data()[k + p * (t + q * (s + p * u))];
    if(!reshaped.allFinite()) throw std::invalid_argument("Energy operator must be finite");
    Eigen::JacobiSVD<Matrix> svd(reshaped, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::vector<Product> result;
    for(Index channel = 0; channel < svd.singularValues().size(); ++channel) {
        const double scale = std::sqrt(svd.singularValues()(channel));
        if(scale == 0.) continue;
        Product product{Matrix(p, p), Matrix(q, q)};
        for(Index i = 0; i < p * p; ++i) product.left.data()[i] = scale * svd.matrixU()(i, channel);
        for(Index i = 0; i < q * q; ++i) product.right.data()[i] = scale * svd.matrixV()(i, channel);
        result.push_back(std::move(product));
    }
    return result;
}

template <bool Track, typename Patch>
auto evaluate(const Patch& patch, const std::vector<Product>& operators, Opts::Bond bond, Index block_size)
{
    std::array<MatrixHandle, 4> corners{corner<Track>(patch.c0), corner<Track>(patch.c1), corner<Track>(patch.c2), corner<Track>(patch.c3, true)};
    std::vector<Edge> top, right, bottom, left;
    for(const auto& t : patch.top) top.push_back(edge<Track, 0>(t));
    for(const auto& t : patch.right) right.push_back(edge<Track, 1>(t));
    for(const auto& t : patch.bottom) bottom.push_back(edge<Track, 2>(t));
    for(const auto& t : patch.left) left.push_back(edge<Track, 3>(t));
    auto site_corner = [patch, corners, top, right, bottom, left, block_size]<bool Inner>(int k, const Matrix& op) {
        const std::array<int, 4> sites{0, patch.width - 1, patch.width * patch.height - 1, patch.width * (patch.height - 1)};
        const std::array<Edge, 4> previous{left[0], top.back(), right.back(), bottom[0]};
        const std::array<Edge, 4> next{top[0], right[0], bottom.back(), left.back()};
        const int site = sites[k];
        const auto& ket = patch.ket[site]; const auto& bra = patch.bra[site];
        std::array<Index, 4> q{dimension<0>(ket) * dimension<0>(bra), dimension<1>(ket) * dimension<1>(bra),
                               dimension<2>(ket) * dimension<3>(bra), dimension<3>(ket) * dimension<4>(bra)};
        auto layer = bulk<Inner>(ket, bra, op, k);
        if(patch.width == 2 && patch.height == 2)
            return enlarged<Inner>(corners[k], previous[k], next[k], layer, q[(k + 2) % 4], q[(k + 3) % 4], block_size);
        const Edge& closing = k == 0 ? bottom[0] : k == 1 ? left[0] : k == 2 ? top.back() : right.back();
        return half_site<Inner>(corners[k], previous[k], next[k], corners[(k + 3) % 4], closing, layer, q[(k + 2) % 4], block_size);
    };
    using Result = std::conditional_t<Track, stan::math::var, double>;
    Result norm, numerator = 0.;
    if(patch.width == 2 && patch.height == 2) {
        std::array<MatrixHandle, 4> identity;
        const std::array<int, 4> sites{0, 1, 3, 2};
        for(int k = 0; k < 4; ++k) {
            const Index d = dimension<4>(patch.ket[sites[k]]);
            identity[k] = site_corner.template operator()<Track>(k, Matrix::Identity(d, d));
        }
        norm = trace_ring<Track>(identity);
        const int first = bond == Opts::Bond::D1 ? 0 : 1, second = first + 2;
        for(const auto& product : operators) numerator += checkpoint<Track>([site_corner, identity, product, first, second]<bool Inner>() {
            auto ring = identity;
            ring[first] = site_corner.template operator()<Inner>(first, product.left);
            ring[second] = site_corner.template operator()<Inner>(second, product.right);
            return trace_ring<Inner>(ring);
        });
    } else {
        const int first = bond == Opts::Bond::H ? 0 : 1, second = (first + 2) % 4;
        const Index p = dimension<4>(patch.ket[0]), r = dimension<4>(patch.ket[1]);
        const Index q = bond == Opts::Bond::H ? dimension<2>(patch.ket[0]) * dimension<3>(patch.bra[0])
                                              : dimension<3>(patch.ket[0]) * dimension<4>(patch.bra[0]);
        auto a = site_corner.template operator()<Track>(first, Matrix::Identity(p, p));
        auto b = site_corner.template operator()<Track>(second, Matrix::Identity(r, r));
        norm = trace_pair<Track>(a, b, q);
        for(const auto& product : operators) numerator += checkpoint<Track>([site_corner, product, first, second, q]<bool Inner>() {
            auto a = site_corner.template operator()<Inner>(first, product.left);
            auto b = site_corner.template operator()<Inner>(second, product.right);
            return trace_pair<Inner>(a, b, q);
        });
    }
    const double norm_value = [&]() { if constexpr(Track) return norm.val(); else return norm; }();
    if(!std::isfinite(norm_value) || std::abs(norm_value) <= std::numeric_limits<double>::min()) throw std::runtime_error("Vanishing sliced energy normalization");
    return numerator / norm;
}
} // namespace Xped::detail::energy_slicing

namespace Xped::detail {
template <bool Track, typename Env, typename Operator>
auto sliced_energy(const Env& env, const Operator& op, int x, int y, Opts::Bond bond, std::size_t block_size = 8)
{
    if(block_size == 0) throw std::invalid_argument("Energy block size must be positive");
    using Symmetry = typename Operator::Symmetry;
    energy_slicing::Patch<Symmetry, Track> patch;
    patch.width = bond == Opts::Bond::V ? 1 : 2;
    patch.height = bond == Opts::Bond::H ? 1 : 2;
    const int w = patch.width, h = patch.height;
    patch.c0 = env.C1s(x - 1, y - 1); patch.c1 = env.C2s(x + w, y - 1);
    patch.c2 = env.C3s(x + w, y + h); patch.c3 = env.C4s(x - 1, y + h);
    for(int i = 0; i < w; ++i) { patch.top.push_back(env.T1s(x + i, y - 1)); patch.bottom.push_back(env.T3s(x + i, y + h)); }
    for(int j = 0; j < h; ++j) { patch.right.push_back(env.T2s(x + w, y + j)); patch.left.push_back(env.T4s(x - 1, y + j)); }
    for(int j = 0; j < h; ++j) for(int i = 0; i < w; ++i) {
        patch.ket.push_back(env.A->As(x + i, y + j)); patch.bra.push_back(env.A->Adags(x + i, y + j));
    }
    const auto operators = energy_slicing::products(op);
    return energy_slicing::checkpoint<Track>([patch, operators, bond, block_size]<bool Inner>() {
        return energy_slicing::evaluate<Inner>(patch, operators, bond, static_cast<energy_slicing::Index>(block_size));
    });
}
} // namespace Xped::detail
#endif
#endif
