#include "Xped/Interfaces/PlainInterface_Eigen_impl.hpp"

#include <Eigen/SVD>

namespace Xped {

using Indextype = Eigen::Index;

template <typename Scalar>
using MType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
template <typename Scalar>
using cMType = const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar, std::size_t Rank>
using TType = Eigen::Tensor<Scalar, Rank>;

template <typename Scalar>
using VType = Eigen::Array<Scalar, Eigen::Dynamic, 1>;

template <std::size_t Rank, typename Derived, typename ClebschScalar, typename Scalar>
void PlainInterface::set_block_from_tensor(Eigen::MatrixBase<Derived>& M,
                                           const Indextype& row,
                                           const Indextype& col,
                                           const Indextype& rows,
                                           const Indextype& cols,
                                           const ClebschScalar& scale,
                                           const TType<Scalar, Rank>& T)
{
    M.block(row, col, rows, cols) = scale * Eigen::Map<cMType<Scalar>>(T.data(), rows, cols);
}

template <std::size_t Rank, typename Derived, typename ClebschScalar, typename Scalar>
void PlainInterface::add_to_block_from_tensor(Eigen::MatrixBase<Derived>& M,
                                              const Indextype& row,
                                              const Indextype& col,
                                              const Indextype& rows,
                                              const Indextype& cols,
                                              const ClebschScalar& scale,
                                              const TType<Scalar, Rank>& T)
{
    M.block(row, col, rows, cols) += scale * Eigen::Map<cMType<Scalar>>(T.data(), rows, cols);
}

template <std::size_t Rank, typename Derived, typename ClebschScalar, typename Scalar>
void PlainInterface::set_block_from_tensor(Eigen::MatrixBase<Derived>&& M,
                                           const Indextype& row,
                                           const Indextype& col,
                                           const Indextype& rows,
                                           const Indextype& cols,
                                           const ClebschScalar& scale,
                                           const TType<Scalar, Rank>& T)
{
    M.block(row, col, rows, cols) = scale * Eigen::Map<cMType<Scalar>>(T.data(), rows, cols);
}

template <std::size_t Rank, typename Derived, typename ClebschScalar, typename Scalar>
void PlainInterface::add_to_block_from_tensor(Eigen::MatrixBase<Derived>&& M,
                                              const Indextype& row,
                                              const Indextype& col,
                                              const Indextype& rows,
                                              const Indextype& cols,
                                              const ClebschScalar& scale,
                                              const TType<Scalar, Rank>& T)
{
    M.block(row, col, rows, cols) += scale * Eigen::Map<cMType<Scalar>>(T.data(), rows, cols);
}

template <std::size_t Rank, typename Derived>
TType<typename Derived::Scalar, Rank> PlainInterface::tensor_from_matrix_block(const Eigen::MatrixBase<Derived>& M,
                                                                               const Indextype& row,
                                                                               const Indextype& col,
                                                                               const Indextype& rows,
                                                                               const Indextype& cols,
                                                                               const std::array<Indextype, Rank>& dims)
{
    MType<typename Derived::Scalar> submatrix = M.block(row, col, rows, cols);
    cMapTType<typename Derived::Scalar, Rank> tensorview = cMap(submatrix.data(), dims);
    return construct<typename Derived::Scalar, Rank>(tensorview);
}

template <typename Scalar>
void PlainInterface::vec_diff(const Eigen::Matrix<Scalar, -1, 1>& vec, MType<Scalar>& res)
{
    res = -(vec.replicate(1, vec.rows()).rowwise() - vec.transpose());
    // res.resize(vec.size(), vec.size());
    // res.diagonal().setZero();
    // for(auto i = 0; i < vec.size(); ++i) {
    //     for(auto j = 0; j < i; ++j) {
    //         res(i, j) = vec(j) - vec(i);
    //         res(j, i) = -res(i, j);
    //     }
    // }
}

template <typename Scalar>
void PlainInterface::vec_add(const Eigen::Matrix<Scalar, -1, 1>& vec, MType<Scalar>& res)
{
    res = vec.replicate(1, vec.rows()).rowwise() + vec.transpose();
    // res.resize(vec.size(), vec.size());
    // for(auto i = 0; i < vec.size(); ++i) {
    //     for(auto j = 0; j <= i; ++j) {
    //         res(i, j) = vec(i) + vec(j);
    //         if(i != j) res(j, i) = res(i, j);
    //     }
    // }
}

template <typename Derived>
void PlainInterface::diagonal_head_matrix_to_vector(VType<typename Derived::Scalar>& V, const Eigen::MatrixBase<Derived>& M, const Indextype& n_elems)
{
    V = M.diagonal().head(n_elems);
}

template <typename Derived>
std::tuple<MType<typename Derived::Scalar>, VType<typename Derived::RealScalar>, MType<typename Derived::Scalar>>
PlainInterface::svd(const Eigen::MatrixBase<Derived>& M)
{
#ifdef XPED_DONT_USE_BDCSVD
    Eigen::JacobiSVD<MType<typename Derived::Scalar>, Eigen::ComputeThinU | Eigen::ComputeThinV> Jack; // standard SVD
#else
    Eigen::BDCSVD<MType<typename Derived::Scalar>, Eigen::ComputeThinU | Eigen::ComputeThinV>
        Jack; // Divide-and-conquer SVD; LAPACK gesdd when LAPACKE is enabled.
#endif

    Jack.compute(M);
    return std::make_tuple(Jack.matrixU(), Jack.singularValues(), Jack.matrixV().adjoint());
}

template <typename Scalar>
MType<Scalar> PlainInterface::vec_to_diagmat(const VType<Scalar>& V)
{
    return V.matrix().asDiagonal();
}

} // namespace Xped

#if __has_include("PlainInterface_Eigen_impl.gen.cpp")
#    include "PlainInterface_Eigen_impl.gen.cpp"
#endif
