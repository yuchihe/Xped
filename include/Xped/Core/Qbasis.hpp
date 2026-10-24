#ifndef QBASIS_H_
#define QBASIS_H_

/// \cond
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "tabulate/table.hpp"
// #include "tabulate/tabulate.hpp"

#include "yas/serialize.hpp"
#include "yas/std_types.hpp"
/// \endcond

#include "Xped/Core/Basis.hpp"
#include "Xped/Core/FusionTree.hpp"
#include "Xped/Core/allocators/HeapPolicy.hpp"
#include "Xped/Hash/hash.hpp"

namespace Xped {

/** \class Qbasis
 *
 * \ingroup Tensors
 * \describe_Symmetry
 *
 * This class is a container like class for a basis of a Hilbert space in which global symmetries are present.
 * For each irreducible representation (irrep) of the global symmetry (for each quantum number),
 * the states of the Hilbert states that transforms under that irrep are collected together in a plain Basis object.
 *
 * One central function is the combine() method, which combine two instances of Qbasis to the tensor product basis,
 * already proper sorted into irreps.
 *
 */
template <typename Symmetry, std::size_t depth = 1, typename AllocationPolicy = HeapPolicy>
class Qbasis
{
    template <typename Symmetry_, std::size_t depth_, typename AllocationPolicy_>
    friend class Qbasis;

    typedef typename Symmetry::qType qType;

    using ContainerType =
        std::vector<std::tuple<qType, size_t, Basis>, typename AllocationPolicy::template Allocator<std::tuple<qType, std::size_t, Basis>>>;
    template <std::size_t tree_depth>
    using TreeVector = std::vector<FusionTree<tree_depth, Symmetry>, typename AllocationPolicy::template Allocator<FusionTree<tree_depth, Symmetry>>>;
    using TreeType = std::unordered_map<
        qType,
        TreeVector<depth>,
        std::hash<qType>,
        std::equal_to<qType>,
        typename AllocationPolicy::template Allocator<
            std::pair<const qType,
                      std::vector<FusionTree<depth, Symmetry>, typename AllocationPolicy::template Allocator<FusionTree<depth, Symmetry>>>>>>;

    // Match FusionTree::operator==, which deliberately ignores duality outside
    // pedantic builds and ignores multiplicities for multiplicity-free symmetries.
    struct OffsetTreeHash
    {
        std::size_t operator()(const FusionTree<depth, Symmetry>& tree) const
        {
            std::size_t seed = 0;
            boost::hash_combine(seed, tree.dims);
            if constexpr(not Symmetry::ALL_IS_TRIVIAL) {
                boost::hash_combine(seed, tree.q_uncoupled);
                boost::hash_combine(seed, tree.q_coupled);
                boost::hash_combine(seed, tree.q_intermediates);
                if constexpr(Symmetry::ANY_HAS_MULTIPLICITIES) { boost::hash_combine(seed, tree.multiplicities); }
#ifdef XPED_PEDANTIC_ASSERTS
                boost::hash_combine(seed, tree.IS_DUAL);
#endif
            }
            return seed;
        }
    };
    using TreeOffsets = std::pair<std::size_t, std::size_t>;
    using TreeOffsetMap = std::unordered_map<FusionTree<depth, Symmetry>,
                                           TreeOffsets,
                                           OffsetTreeHash,
                                           std::equal_to<FusionTree<depth, Symmetry>>,
                                           typename AllocationPolicy::template Allocator<std::pair<const FusionTree<depth, Symmetry>, TreeOffsets>>>;
    using OffsetMap = std::unordered_map<qType,
                                         TreeOffsetMap,
                                         std::hash<qType>,
                                         std::equal_to<qType>,
                                         typename AllocationPolicy::template Allocator<std::pair<const qType, TreeOffsetMap>>>;

public:
    /**Does nothing.*/
    Qbasis() = default;
    Qbasis(const Qbasis&) = default;
    Qbasis(Qbasis&&) = default;

    Qbasis& operator=(Qbasis other)
    {
        // An earlier allTrees() reference still names this object's map after
        // assignment; replacing the contents cannot make caching safe again.
        const bool cache_offsets = cache_tree_offsets_ and other.cache_tree_offsets_;
        using std::swap;
        swap(data_, other.data_);
        swap(curr_dim, other.curr_dim);
        swap(trees, other.trees);
        swap(tree_offsets_, other.tree_offsets_);
        swap(IS_SORTED_, other.IS_SORTED_);
        swap(CONJ, other.CONJ);
        cache_tree_offsets_ = cache_offsets;
        if(not cache_tree_offsets_) { tree_offsets_.clear(); }
        return *this;
    }

    /**
     * Inserts all quantum numbers in the Container \p qins with constant dimension \p dim into the basis.
     */
    template <typename Container>
    Qbasis(const Container& qins, const size_t& dim)
    {
        for(const auto& qin : qins) { push_back(qin, dim); }
    }

    static Qbasis<Symmetry, 1, AllocationPolicy> TrivialBasis();

    ///\{
    /**Returns the number of (reduced) basis states.*/
    inline std::size_t dim() const
    {
        std::size_t out = 0;
        for(const auto& [qVal, num, plain] : data_) { out += plain.dim(); }
        return out;
    }

    /**
     * Returns the full number of basis states.
     * If irreps has internal states, the basis states transforming under this irreps are multiplied by this degeneracy.
     */
    inline std::size_t fullDim() const
    {
        std::size_t out = 0;
        for(const auto& [qVal, num, plain] : data_) { out += plain.dim() * Symmetry::degeneracy(qVal); }
        return out;
    }

    /**Returns the largest state sector.*/
    std::size_t largestSector() const;

    /**Returns the number of quantum numbers (irreps) contained in this basis.*/
    inline std::size_t Nq() const { return data_.size(); }
    ///\}

    /**
     * Returns the quantum number of the state with number \p num.
     */
    qType getQ(const size_t& num) const;

    ///\{
    inline std::vector<qType> qs() const
    {
        std::vector<qType> out;
        out.reserve(data_.size());
        for(auto triple : data_) { out.push_back(std::get<0>(triple)); }
        return out;
    }
    inline std::unordered_set<qType> unordered_qs() const
    {
        std::unordered_set<qType> out;
        for(auto triple : data_) { out.insert(std::get<0>(triple)); }
        return out;
    }
    ///\}

    ///\{
    inline std::vector<std::size_t> dims() const
    {
        std::vector<std::size_t> out;
        out.reserve(data_.size());
        for(auto triple : data_) { out.push_back(std::get<2>(triple).dim()); }
        return out;
    }
    ///\}

    ///\{
    /**Checks whether states with quantum number \p q are in the basis. Returns true if the state is present.*/
    bool IS_PRESENT(const qType& q) const { return cfind(q) != data_.end(); }
    /**Checks whether states with quantum number \p q are not in the basis. Returns true if the state is absent.*/
    bool NOT_PRESENT(const qType& q) const { return !IS_PRESENT(q); }
    ///\}

    ///\{
    /**Computes the number within the sector from the total number of the state*/
    size_t inner_num(const size_t& outer_num) const;
    /**Computes the sector size of a state with number \p num_in.*/
    size_t inner_dim(const size_t& num_in) const;
    /**Computes the sector size of a state with quantum number \p q.*/
    size_t inner_dim(const qType& q) const;
    /**Computes the total number with which the sector with quantum number \p q begins.*/
    size_t outer_num(const qType& q) const;
    /**Computes the total number with which the sector with quantum number \p q begins and takes into account the dimension of each irrep.*/
    size_t full_outer_num(const qType& q) const;
    /**Returns the index of a given quantum number.*/
    size_t index(const qType& q) const;
    ///\}

    /**Inverse function to `index(const qType&)`.*/
    qType qn(std::size_t index) const
    {
        assert(index < data_.size());
        return std::get<0>(data_[index]);
    };

    void remove(const qType& q);

    inline qType maxQ() const
    {
        auto qs = this->qs();
        return *std::max_element(qs.begin(), qs.end());
    }
    inline qType minQ() const
    {
        auto qs = this->qs();
        return *std::min_element(qs.begin(), qs.end());
    }
    // template<typename Iterator>
    // void remove(Iterator& begin, Iterator& end);

    void truncate(const std::unordered_set<qType>& qs, const std::size_t& M);

    size_t leftOffset(const FusionTree<depth, Symmetry>& tree, const std::array<size_t, depth>& plain = std::array<std::size_t, depth>()) const;
    size_t rightOffset(const FusionTree<depth, Symmetry>& tree, const std::array<size_t, depth>& plain = std::array<std::size_t, depth>()) const;

    /**Insert the quantum number \p q with dimension \p inner_dim into the basis.*/
    void push_back(const qType& q, const size_t& inner_dim);

    void push_back(const qType& q, const size_t& inner_dim, const TreeVector<depth>& tree);

    void setRandom(const std::size_t& fullSize, const std::size_t& max_sectorSize = 5ul);

    /**Completely clear the basis.*/
    inline void clear()
    {
        data_.clear();
        trees.clear();
        tree_offsets_.clear();
        curr_dim = 0;
    }

    /**
     * Returns the tensor product basis, already properly sorted with respect to the resulting irreps.
     * This function also saves the history of the combination process for later use. See leftOffset() and rightOffset().
     */
    Qbasis<Symmetry, depth + 1, AllocationPolicy> combine(const Qbasis<Symmetry, 1, AllocationPolicy>& other, bool CONJ = false) const;

    /**Adds two bases together.*/
    Qbasis<Symmetry, depth, AllocationPolicy> add(const Qbasis<Symmetry, depth, AllocationPolicy>& other) const;

    std::pair<std::vector<std::size_t>, Qbasis<Symmetry, depth, AllocationPolicy>> shift(qType qshift) const;

    /**Returns the intersection of this and \p other.*/
    Qbasis<Symmetry, depth, AllocationPolicy> intersection(const Qbasis<Symmetry, depth, AllocationPolicy>& other) const;

    Qbasis<Symmetry, depth, AllocationPolicy> conj() const;

    Qbasis<Symmetry, 1, AllocationPolicy> forgetHistory() const;

    /**Prints the basis.*/
    std::string print() const;

    /**Prints rough infob.*/
    std::string info() const;

    /**Prints the trees.*/
    std::string printTrees() const;

    bool operator==(const Qbasis<Symmetry, depth, AllocationPolicy>& other) const;

    inline auto begin() { return data_.begin(); }
    inline auto end() { return data_.end(); }

    inline auto cbegin() const { return data_.cbegin(); }
    inline auto cend() const { return data_.cend(); }

    /**Swaps with another Qbasis.*/
    // void swap (Qbasis<Symmetry,depth> &other) { this->data.swap(other.data()); }

    std::vector<std::size_t> sort();

    inline bool IS_SORTED() const { return IS_SORTED_; }
    inline bool IS_CONJ() const { return CONJ; }

    inline void SET_CONJ() { CONJ = true; }

    inline const auto& tree(const qType& q) const
    {
        auto it = trees.find(q);
        assert(it != trees.end());
        return it->second;
    }

    inline TreeType& allTrees()
    {
        // The caller can retain this reference and mutate again after an offset
        // lookup. Keep the original scan path for this object's remaining life.
        cache_tree_offsets_ = false;
        tree_offsets_.clear();
        return trees;
    }

    template <typename Ar>
    void serialize(Ar& ar) const
    {
        ar& YAS_OBJECT_NVP("Qbasis", ("data", data_), ("curr_dim", curr_dim), ("trees", trees), ("IS_SORTED", IS_SORTED_), ("CONJ", CONJ));
    }

    template <typename Ar>
    void serialize(Ar& ar)
    {
        ar& YAS_OBJECT_NVP("Qbasis", ("data", data_), ("curr_dim", curr_dim), ("trees", trees), ("IS_SORTED", IS_SORTED_), ("CONJ", CONJ));
        // Derived metadata is rebuilt, not serialized, to keep archive compatibility.
        rebuild_tree_offsets();
    }

private:
    typename ContainerType::const_iterator cfind(const qType& q) const;

    typename ContainerType::iterator find(const qType& q);

    // vector with entry: {Quantumnumber (QN), state number of the first plain state for this QN, all plain states for this QN in a Basis object.}
    //[{q1,0,plain_q1}, {q2,dim(plain_q1),plain_q2}, {q3,dim(plain_q1)+dim(plain_q2),plain_q3}, ..., {qi, sum_j^(i-1)dim(plain_qj), plain_qi}]
    ContainerType data_{};
    std::size_t curr_dim = 0;
    TreeType trees{};
    OffsetMap tree_offsets_{};
    bool cache_tree_offsets_ = true;

    // Eager updates keep concurrent const offset queries read-only.
    void rebuild_tree_offsets(const qType& q);
    void rebuild_tree_offsets();

    bool IS_SORTED_ = false;
    bool CONJ = false;
};

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
std::ostream& operator<<(std::ostream& os, const Qbasis<Symmetry, depth, AllocationPolicy>& basis)
{
    os << basis.print();
    return os;
}

} // namespace Xped

#ifndef XPED_COMPILED_LIB
#    include "Core/Qbasis.cpp"
#endif

#endif
