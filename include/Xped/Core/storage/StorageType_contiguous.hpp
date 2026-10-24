#ifndef XPED_STORAGE_TYPE_CONTIGOUS_HPP_
#define XPED_STORAGE_TYPE_CONTIGOUS_HPP_

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"

#include "Xped/Interfaces/PlainInterface.hpp"
#include "Xped/Core/Qbasis.hpp"
#include "Xped/Core/TensorHelper.hpp"

namespace Xped {

template <typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, typename AllocationPolicy>
struct StorageType
{
    using MatrixType = PlainInterface::MType<Scalar>;
    using MapMatrixType = PlainInterface::MapMType<Scalar>;
    using cMapMatrixType = PlainInterface::cMapMType<Scalar>;
    using qType = typename Symmetry::qType;

    using MatrixReturnType = MapMatrixType;
    using ConstMatrixReturnType = cMapMatrixType;

private:
    using DictType = std::unordered_map<qType,
                                        std::size_t,
                                        std::hash<qType>,
                                        std::equal_to<qType>,
                                        typename AllocationPolicy::template Allocator<std::pair<const qType, std::size_t>>>;

public:
    StorageType() = default;

    StorageType(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank> basis_domain,
                const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank> basis_codomain,
                const mpi::XpedWorld& world = mpi::getUniverse())
        : m_uncoupled_domain(basis_domain)
        , m_uncoupled_codomain(basis_codomain)
        , m_world(world)
    {
        m_domain = internal::build_FusionTree(m_uncoupled_domain);
        m_codomain = internal::build_FusionTree(m_uncoupled_codomain);
        // uninitialized_resize();
    }

    StorageType(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank> basis_domain,
                const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank> basis_codomain,
                const Scalar* data,
                std::size_t size,
                const mpi::XpedWorld& world = mpi::getUniverse())
        : m_uncoupled_domain(basis_domain)
        , m_uncoupled_codomain(basis_codomain)
        , m_world(world)
    {
        m_domain = internal::build_FusionTree(m_uncoupled_domain);
        m_codomain = internal::build_FusionTree(m_uncoupled_codomain);
        initialized_resize(data, size);
    }

    StorageType(const StorageType<Scalar, Rank, CoRank, Symmetry, AllocationPolicy>& other) = default;
    // {
    //     sector_ = other.derived().sector();
    //     // block_.resize(sector_.size());
    //     storage_ = other.storage();
    //     // for(std::size_t i = 0; i < sector_.size(); i++) { block_[i] = other.derived().block(i); }
    //     dict_ = other.derived().dict();
    //     world_ = other.derived().world();
    //     uncoupled_domain = other.derived().uncoupledDomain();
    //     uncoupled_codomain = other.derived().uncoupledCodomain();
    //     domain = other.derived().coupledDomain();
    //     codomain = other.derived().coupledCodomain();
    // }

    // template <template <typename> typename OtherAllocator>
    // StorageType(const StorageType<Scalar, Rank, CoRank, Symmetry, OtherAllocator>& other);

    static constexpr bool IS_CONTIGUOUS() { return true; }

    void resize()
    {
        clear();
        std::size_t curr = 0;
        for(const auto& [q, dim, plain] : m_domain) {
            if(m_codomain.IS_PRESENT(q)) {
                m_sector.push_back(q);
                m_dict.insert(std::make_pair(q, m_sector.size() - 1));
                m_offsets.push_back(curr);
                m_rows.push_back(plain.dim());
                m_cols.push_back(m_codomain.inner_dim(q));
                curr += m_rows.back() * m_cols.back();
            }
        }
        try {
            m_data.resize(curr);
        } catch(std::bad_alloc& e) {
            fmt::print(stderr, "Failure during allocation of tensor data. Requested elems: {} (mem={}).\n", curr, curr * sizeof(Scalar));
            throw std::bad_alloc();
        }
    }

    void set_data(const Scalar* data, std::size_t size)
    {
        assert(m_sector.size() > 0 and "Block data needs to already set. Use initialized_resize() instead.");
        assert(m_offsets.size() == m_sector.size() and "Corrupted storage.");
        assert(m_dict.size() == m_sector.size() and "Corrupted storage.");
        try {
            m_data.assign(data, data + size);
        } catch(std::bad_alloc& e) {
            fmt::print(stderr, "Failure during allocation of tensor data. Requested elems: {} (mem={}).\n", size, size * sizeof(Scalar));
            throw std::bad_alloc();
        }
    }

    cMapMatrixType block(std::size_t i) const
    {
        assert(i < m_offsets.size());
        return cMapMatrixType(m_data.data() + m_offsets[i], m_rows[i], m_cols[i]);
    }

    MapMatrixType block(std::size_t i)
    {
        assert(i < m_offsets.size());
        return MapMatrixType(m_data.data() + m_offsets[i], m_rows[i], m_cols[i]);
    }

    cMapMatrixType block(qType q) const
    {
        auto it = m_dict.find(q);
        assert(it != m_dict.end());
        return block(it->second);
    }

    MapMatrixType block(qType q)
    {
        auto it = m_dict.find(q);
        assert(it != m_dict.end());
        return block(it->second);
    }

    const DictType& dict() const { return m_dict; }
    DictType& dict() { return m_dict; }

    const std::vector<qType, typename AllocationPolicy::template Allocator<qType>>& sector() const { return m_sector; }
    qType sector(std::size_t i) const { return m_sector[i]; }

    const Scalar* data() const { return m_data.data(); }
    Scalar* data() { return m_data.data(); }

    auto begin() { return m_data.begin(); }
    auto end() { return m_data.end(); }

    const auto cbegin() const { return m_data.cbegin(); }
    const auto cend() const { return m_data.cend(); }

    std::size_t plainSize() const { return m_data.size(); }

    const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank>& uncoupledDomain() const { return m_uncoupled_domain; }
    const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank>& uncoupledCodomain() const { return m_uncoupled_codomain; }

    const Qbasis<Symmetry, Rank, AllocationPolicy>& coupledDomain() const { return m_domain; }
    const Qbasis<Symmetry, CoRank, AllocationPolicy>& coupledCodomain() const { return m_codomain; }

    void push_back(const qType& q, const MatrixType& M)
    {
        if(m_offsets.size() == 0) {
            m_offsets.push_back(0);
        } else {
            m_offsets.push_back(m_offsets.back() + m_rows.back() * m_cols.back());
        }
        m_sector.push_back(q);
        m_dict.insert(std::make_pair(q, m_sector.size() - 1));
        m_rows.push_back(m_domain.inner_dim(q));
        m_cols.push_back(m_codomain.inner_dim(q));
        assert(static_cast<std::size_t>(PlainInterface::rows(M)) == m_rows.back());
        assert(static_cast<std::size_t>(PlainInterface::cols(M)) == m_cols.back());
        m_data.insert(m_data.end(), M.data(), M.data() + M.size());
    }

    void reserve(std::size_t size)
    {
        try {
            m_data.reserve(size);
        } catch(std::bad_alloc& e) {
            fmt::print(stderr, "Failure during allocation of tensor data. Requested elems: {} (mem={}).\n", size, size * sizeof(Scalar));
            throw std::bad_alloc();
        }
    }

    void clear()
    {
        m_data.clear();
        m_sector.clear();
        m_dict.clear();
        m_offsets.clear();
        m_rows.clear();
        m_cols.clear();
    }

    template <typename Ar>
    void serialize(Ar& ar) const
    {
        ar& YAS_OBJECT_NVP("x_storage",
                           ("data", m_data),
                           ("dict", m_dict),
                           ("sectors", m_sector),
                           ("uncoupled_domain", m_uncoupled_domain),
                           ("uncoupled_codomain", m_uncoupled_codomain),
                           ("domain", m_domain),
                           ("codomain", m_codomain),
                           ("world", m_world),
                           ("offsets", m_offsets));
    }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("x_storage",
                           ("data", m_data),
                           ("dict", m_dict),
                           ("sectors", m_sector),
                           ("uncoupled_domain", m_uncoupled_domain),
                           ("uncoupled_codomain", m_uncoupled_codomain),
                           ("domain", m_domain),
                           ("codomain", m_codomain),
                           ("world", m_world),
                           ("offsets", m_offsets));
        // Derived metadata stays out of the archive, preserving existing files.
        rebuild_block_dimensions();
    }

    const mpi::XpedWorld& world() const { return m_world; }

private:
    std::vector<Scalar, typename AllocationPolicy::template Allocator<Scalar>> m_data;

    DictType m_dict; // sector --> number
    std::vector<qType, typename AllocationPolicy::template Allocator<qType>> m_sector;

    std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank> m_uncoupled_domain;
    std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank> m_uncoupled_codomain;
    Qbasis<Symmetry, Rank, AllocationPolicy> m_domain;
    Qbasis<Symmetry, CoRank, AllocationPolicy> m_codomain;

    mpi::XpedWorld m_world;

    std::vector<std::size_t, typename AllocationPolicy::template Allocator<std::size_t>> m_offsets;
    std::vector<std::size_t, typename AllocationPolicy::template Allocator<std::size_t>> m_rows;
    std::vector<std::size_t, typename AllocationPolicy::template Allocator<std::size_t>> m_cols;

    void rebuild_block_dimensions()
    {
        m_rows.clear();
        m_cols.clear();
        m_rows.reserve(m_sector.size());
        m_cols.reserve(m_sector.size());
        for(const auto& q : m_sector) {
            m_rows.push_back(m_domain.inner_dim(q));
            m_cols.push_back(m_codomain.inner_dim(q));
        }
    }

    void initialized_resize(const Scalar* data, std::size_t size)
    {
        try {
            m_data.assign(data, data + size);
        } catch(std::bad_alloc& e) {
            fmt::print(stderr, "Failure during allocation of tensor data. Requested elems: {} (mem={}).\n", size, size * sizeof(Scalar));
            throw std::bad_alloc();
        }

        std::size_t curr = 0;
        for(const auto& [q, dim, plain] : m_domain) {
            if(m_codomain.IS_PRESENT(q)) {
                m_sector.push_back(q);
                m_dict.insert(std::make_pair(q, m_sector.size() - 1));
                m_offsets.push_back(curr);
                m_rows.push_back(plain.dim());
                m_cols.push_back(m_codomain.inner_dim(q));
                curr += m_rows.back() * m_cols.back();
            }
        }
        assert(curr == size and "You specified an incompatible data array for this tensor.");
    }
};

} // namespace Xped

#endif
