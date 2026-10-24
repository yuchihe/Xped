#include <iostream>
#include <numeric>
#include <variant>
#include <vector>

#include "fmt/core.h"

#include "Xped/Symmetry/functions.hpp"
#include "Xped/Util/Random.hpp"

#include "Xped/Symmetry/SU2.hpp"
#include "Xped/Symmetry/U0.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/ZN.hpp"
#include "Xped/Symmetry/S1xS2.hpp"
#include "Xped/Symmetry/CombSym.hpp"

#include "Xped/Core/Qbasis.hpp"

using std::cout;
using std::endl;
using std::size_t;

namespace Xped {

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
Qbasis<Symmetry, 1, AllocationPolicy> Qbasis<Symmetry, depth, AllocationPolicy>::TrivialBasis()
{
    Qbasis<Symmetry, 1, AllocationPolicy> out;
    out.push_back(Symmetry::qvacuum(), 1);
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
void Qbasis<Symmetry, depth, AllocationPolicy>::push_back(const qType& q, const size_t& inner_dim)
{
    assert(depth == 1 or depth == 0);
    if(inner_dim == 0) { return; }
    auto it = find(q);
    if(it == data_.end()) // insert quantum number if it is not there (sorting is lost then)
    {
        Basis plain_basis(inner_dim);
        auto entry = std::make_tuple(q, curr_dim, plain_basis);
        if constexpr(depth == 1) {
            FusionTree<1, Symmetry> trivial;
            trivial.q_uncoupled[0] = q;
            trivial.q_coupled = q;
            trivial.IS_DUAL[0] = CONJ;
            trivial.dims[0] = inner_dim;
            trivial.dim = inner_dim;
            TreeVector<1> tree(1, trivial);
            trees.insert(std::make_pair(q, tree));
        } else if constexpr(depth == 0) {
            FusionTree<0, Symmetry> trivial;
            trivial.dim = 1;
            trivial.q_coupled = q;
            TreeVector<0> tree(1, trivial);
            trees.insert(std::make_pair(q, tree));
        }
        data_.push_back(entry);
        IS_SORTED_ = false;
    } else // append to quantumnumber if it is there
    {
        std::get<2>(*it) = std::get<2>(*it).add(Basis(inner_dim));
        for(auto loop = ++it; loop != data_.end(); ++loop) { std::get<1>(*loop) += inner_dim; }
        for(auto& [p, tree] : trees) {
            if(p != q) { continue; }
            tree[0].dim += inner_dim;
            tree[0].dims[0] += inner_dim;
        }
    }
    curr_dim += inner_dim;
    rebuild_tree_offsets(q);
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
void Qbasis<Symmetry, depth, AllocationPolicy>::push_back(const qType& q, const size_t& inner_dim, const TreeVector<depth>& tree)
{
    auto it = find(q);
    if(it == data_.end()) // insert quantum number if it is not there (sorting is lost then)
    {
        Basis plain_basis(inner_dim);
        auto entry = std::make_tuple(q, curr_dim, plain_basis);
        trees.insert(std::make_pair(q, tree));
        data_.push_back(entry);
        IS_SORTED_ = false;
    } else // append to quantumnumber if it is there
    {
        assert(false);
    }
    curr_dim += inner_dim;
    rebuild_tree_offsets(q);
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
void Qbasis<Symmetry, depth, AllocationPolicy>::setRandom(const std::size_t& fullSize, const std::size_t& max_sectorSize)
{
    clear();
    assert(depth == 1);
    while(fullDim() < fullSize) {
        qType q = Symmetry::random_q();
        std::size_t inner_dim = static_cast<std::size_t>(random::threadSafeRandUniform<int, int>(1, std::min(max_sectorSize, fullSize - fullDim())));
        if((fullDim() + Symmetry::degeneracy(q) * inner_dim) <= fullSize) { push_back(q, inner_dim); }
    }
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
std::size_t Qbasis<Symmetry, depth, AllocationPolicy>::largestSector() const
{
    std::size_t out = 0;
    for(const auto& entry : data_) {
        auto [qVal, num, plain] = entry;
        if(plain.dim() > out) { out = plain.dim(); }
    }
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
typename std::vector<std::tuple<typename Symmetry::qType, size_t, Basis>,
                     typename AllocationPolicy::template Allocator<std::tuple<typename Symmetry::qType, size_t, Basis>>>::const_iterator
Qbasis<Symmetry, depth, AllocationPolicy>::cfind(const qType& q) const
{
    auto it = std::find_if(data_.cbegin(), data_.cend(), [&q](const std::tuple<qType, size_t, Basis>& entry) { return std::get<0>(entry) == q; });
    return it;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
typename std::vector<std::tuple<typename Symmetry::qType, size_t, Basis>,
                     typename AllocationPolicy::template Allocator<std::tuple<typename Symmetry::qType, size_t, Basis>>>::iterator
Qbasis<Symmetry, depth, AllocationPolicy>::find(const qType& q)
{
    auto it = std::find_if(data_.begin(), data_.end(), [&q](const std::tuple<qType, size_t, Basis>& entry) { return std::get<0>(entry) == q; });
    return it;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
typename Symmetry::qType Qbasis<Symmetry, depth, AllocationPolicy>::getQ(const size_t& num_in) const
{
    assert(num_in < dim() and "The number is larger than the size of this basis.");
    for(const auto& [qVal, num, basis] : data_) {
        if(num_in < num + basis.dim()) { return qVal; }
    }
    assert(false and "Something went wrong in Qbasis::find(size_t num_in)");
    return Symmetry::qvacuum();
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
size_t Qbasis<Symmetry, depth, AllocationPolicy>::inner_num(const size_t& outer_num) const
{
    assert(outer_num < dim() and "The number is larger than the size of this basis.");
    for(const auto& [qVal, num, plain] : data_) {
        if(outer_num >= num and outer_num < num + plain.dim()) { return outer_num - num; }
    }
    assert(false and "Something went wrong in Qbasis::inner_num");
    return 0;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
size_t Qbasis<Symmetry, depth, AllocationPolicy>::inner_dim(const qType& q) const
{
    if(auto it = cfind(q); it == data_.end()) {
        return 0;
    } else {
        return std::get<2>(*it).dim();
    }
    return 0;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
size_t Qbasis<Symmetry, depth, AllocationPolicy>::outer_num(const qType& q) const
{
    auto it = cfind(q);
    if(it == data_.end()) { assert(false and "The quantum number is not in the basis"); }
    return std::get<1>(*it);
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
size_t Qbasis<Symmetry, depth, AllocationPolicy>::full_outer_num(const qType& q) const
{
    size_t out = 0ul;
    for(const auto& [p, num, plain] : data_) {
        if(q == p) {
            return out;
        } else {
            out += plain.dim() * Symmetry::degeneracy(p);
        }
    }
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
size_t Qbasis<Symmetry, depth, AllocationPolicy>::index(const qType& q) const
{
    auto it = cfind(q);
    assert(it != data_.end() and "The quantum number is not in the basis");
    return std::distance(data_.begin(), it);
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
size_t Qbasis<Symmetry, depth, AllocationPolicy>::inner_dim(const size_t& num_in) const
{
    for(const auto& elem : data_) {
        auto [qVal, num, plain] = elem;
        if(num == num_in) { return plain.dim(); }
    }
    assert(1 != 1 and "This number is not in the basis");
    return 0;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
void Qbasis<Symmetry, depth, AllocationPolicy>::rebuild_tree_offsets(const qType& q)
{
    // A one-leg sector has only one tree; scanning it is cheaper than caching it.
    if constexpr(depth < 2) { return; }
    tree_offsets_.erase(q);
    if(not cache_tree_offsets_) { return; }
    const auto it = trees.find(q);
    if(it == trees.end()) { return; }
    TreeOffsetMap offsets;
    offsets.reserve(it->second.size());
    std::size_t total = 0;
    for(const auto& tree : it->second) {
        auto entry = offsets.try_emplace(tree, TreeOffsets{total, 0}).first;
        // Accumulate all equal entries so even duplicate histories retain the
        // original leftOffset (first match) and rightOffset (skip equal entries).
        entry->second.second += tree.dim;
        total += tree.dim;
    }
    for(auto& [tree, offset] : offsets) { offset.second = total - offset.first - offset.second; }
    tree_offsets_.emplace(q, std::move(offsets));
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
void Qbasis<Symmetry, depth, AllocationPolicy>::rebuild_tree_offsets()
{
    tree_offsets_.clear();
    if constexpr(depth < 2) { return; }
    if(not cache_tree_offsets_) { return; }
    for(const auto& [q, history] : trees) { rebuild_tree_offsets(q); }
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
size_t Qbasis<Symmetry, depth, AllocationPolicy>::leftOffset(const FusionTree<depth, Symmetry>& tree, const std::array<size_t, depth>& plain) const
{
    assert(trees.size() == data_.size() and "The history for this basis is not defined properly");
    if constexpr(depth >= 2) {
        const auto sector = tree_offsets_.find(tree.q_coupled);
        if(sector != tree_offsets_.end()) {
            const auto offset = sector->second.find(tree);
            if(offset != sector->second.end()) { return offset->second.first; }
        }
    }
    auto it = trees.find(tree.q_coupled);
    assert(it != trees.end() and "The history for this basis is not defined properly");
    size_t out = 0;

    for(const auto& i : it->second) {
        if(i != tree) { out += i.dim; }
        if(i == tree) {
            [[maybe_unused]] auto tmp = plain.size(); // supresses gcc warning
            // maybe some code for plain
            break;
        }
    }
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
size_t Qbasis<Symmetry, depth, AllocationPolicy>::rightOffset(const FusionTree<depth, Symmetry>& tree, const std::array<size_t, depth>& plain) const
{
    assert(trees.size() == data_.size() and "The history for this basis is not defined properly");

    if constexpr(depth >= 2) {
        const auto sector = tree_offsets_.find(tree.q_coupled);
        if(sector != tree_offsets_.end()) {
            const auto offset = sector->second.find(tree);
            if(offset != sector->second.end()) { return offset->second.second; }
        }
    }
    auto it = trees.find(tree.q_coupled);
    assert(it != trees.end() and "The history for this basis is not defined properly");
    size_t out = 0;
    bool SWITCH = false;

    for(const auto& i : it->second) {
        if(i != tree and SWITCH == true) { out += i.dim; }
        if(i == tree) {
            plain.size(); // supresses gcc warning
            // some code for plain size
            SWITCH = true;
        }
    }
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
void Qbasis<Symmetry, depth, AllocationPolicy>::remove(const qType& Q)
{
    auto it = find(Q);
    if(it != data_.end()) {
        auto dimQ = std::get<2>(*it).dim();
        it = data_.erase(it);
        for(auto loop = it; loop != data_.end(); ++loop) { std::get<1>(*loop) -= dimQ; }
        curr_dim -= dimQ;
    }
    trees.erase(Q);
    tree_offsets_.erase(Q);
    // std::get<2>(*it) = std::get<2>(*it).add(Basis(inner_dim));
    // for (auto& [p,tree] : trees) {
    //         if (p != q) {continue;}
    //         tree[0].dim += inner_dim;
    //         tree[0].dims[0] += inner_dim;
    // }
}

// template<typename Iterator>
// template<typename Symmetry, std::size_t depth>
// void Qbasis<Symmetry,depth>::
// remove (Iterator& begin, Iterator& end)
// {
//         for(auto it=begin, it != end; it++) {
//                 remove(*it);
//         }
// }

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
void Qbasis<Symmetry, depth, AllocationPolicy>::truncate(const std::unordered_set<qType>& qs, const std::size_t& M)
{
    assert(depth == 1 and "Cannot truncate a Qbasis with a depth != 1.");
    if(qs.size() < Nq()) {
        for(const auto& q : this->qs()) {
            if(auto it = qs.find(q); it == qs.end()) { remove(q); }
        }
    }
    if(M < dim()) {
        assert(M >= Nq() and "Cannot truncate basis down and keep at least one state per block");
        size_t D = M / Nq();
        size_t D_remainder = M % Nq();

        for(size_t i = 0; i < this->Nq(); i++) {
            std::size_t Deff = (std::get<0>(data_[i]) == Symmetry::qvacuum()) ? D + D_remainder : D;
            if(std::get<2>(data_[i]).dim() > Deff) {
                std::get<2>(data_[i]) = Basis(Deff);
                auto it = trees.find(std::get<0>(data_[i]));
                assert(it->second.size() == 1);
                it->second[0].dims[0] = Deff;
                it->second[0].dim = Deff;
                rebuild_tree_offsets(std::get<0>(data_[i]));
            }
            auto dims = this->dims();
            std::get<1>(data_[i]) = std::accumulate(dims.begin(), dims.begin() + i, 0ul);
        }
    }
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
std::vector<std::size_t> Qbasis<Symmetry, depth, AllocationPolicy>::sort()
{
    // Only sector order changes here; each sector's tree order and offsets stay valid.
    if(data_.size() < 2) { return std::vector<std::size_t>({0}); }
    std::vector<std::size_t> index_sort(data_.size());
    std::iota(index_sort.begin(), index_sort.end(), 0);
    std::sort(index_sort.begin(), index_sort.end(), [&](std::size_t n1, std::size_t n2) {
        qarray<Symmetry::Nq> q1 = std::get<0>(data_[n1]);
        qarray<Symmetry::Nq> q2 = std::get<0>(data_[n2]);
        if(CONJ) {
            return Symmetry::compare(Symmetry::conj(q1), Symmetry::conj(q2));
        } else {
            return Symmetry::compare(q1, q2);
        }
    });
    auto new_data_ = data_;
    for(std::size_t i = 0; i < data_.size(); i++) { new_data_[i] = data_[index_sort[i]]; }
    std::get<1>(new_data_[0]) = 0;
    for(std::size_t i = 1; i < data_.size(); i++) {
        std::get<1>(new_data_[i]) = 0;
        for(std::size_t j = 0; j < i; j++) { std::get<1>(new_data_[i]) += std::get<2>(new_data_[j]).dim(); }
    }
    data_ = new_data_;
    IS_SORTED_ = true;
    return index_sort;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
bool Qbasis<Symmetry, depth, AllocationPolicy>::operator==(const Qbasis<Symmetry, depth, AllocationPolicy>& other) const
{
    // std::cout << *this << std::endl << other << std::endl;
    // std::cout << "this tree" << std::endl;
    // for(const auto& [q, tree] : this->trees) {
    //     std::cout << "Q=" << q << std::endl;
    //     for(const auto& t : tree) { std::cout << t.draw() << std::endl; }
    //     std::cout << std::endl;
    // }
    // std::cout << "other tree" << std::endl;
    // for(const auto& [q, tree] : other.trees) {
    //     std::cout << "Q=" << q << std::endl;
    //     for(const auto& t : tree) { std::cout << t.draw() << std::endl; }
    //     std::cout << std::endl;
    // }
    // std::cout << std::boolalpha << (this->data_ == other.data_) << ", " << (this->trees == other.trees) << std::endl;
    return (this->data_ == other.data_) and (this->trees == other.trees);
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
Qbasis<Symmetry, depth, AllocationPolicy> Qbasis<Symmetry, depth, AllocationPolicy>::add(const Qbasis<Symmetry, depth, AllocationPolicy>& other) const
{
    std::unordered_set<qType> uniqueController;
    Qbasis<Symmetry, depth, AllocationPolicy> out;
    for(const auto& [q1, num1, plain1] : this->data_) {
        auto it_other = std::find_if(
            other.data_.begin(), other.data_.end(), [q1 = q1](std::tuple<qType, size_t, Basis> entry) { return std::get<0>(entry) == q1; });
        if(it_other != other.data_.end()) {
            out.push_back(q1, plain1.add(std::get<2>(*it_other)).dim());
        } else {
            out.push_back(q1, plain1.dim());
        }
    }

    for(const auto& [q2, num2, plain2] : other.data_) {
        auto it_this =
            std::find_if(data_.begin(), data_.end(), [q2 = q2](std::tuple<qType, size_t, Basis> entry) { return std::get<0>(entry) == q2; });
        if(it_this == data_.end()) { out.push_back(q2, plain2.dim()); }
    }
    out.sort();
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
std::pair<std::vector<std::size_t>, Qbasis<Symmetry, depth, AllocationPolicy>> Qbasis<Symmetry, depth, AllocationPolicy>::shift(qType qshift) const
{
    static_assert(depth == 1, "shift() in Qbasis is only for bases without depth.");
    if(qshift == Symmetry::qvacuum()) {
        std::vector<std::size_t> out(this->Nq());
        std::iota(out.begin(), out.end(), 0ul);
        return std::make_pair(out, *this);
    }
    for(auto q = 0u; q < Symmetry::Nq; ++q) {
        if(qshift[q] != Symmetry::qvacuum()[q]) { assert(Symmetry::ABELIAN[q] and "Nontrivial shifts only for Abelian symmetries."); }
    }

    Qbasis<Symmetry, depth, AllocationPolicy> out;
    for(const auto& [q, dim, plain] : data_) { out.push_back(Symmetry::reduceSilent(q, qshift)[0], plain.dim()); }
    auto idx_sort = out.sort();
    return std::make_pair(idx_sort, out);
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
Qbasis<Symmetry, depth, AllocationPolicy>
Qbasis<Symmetry, depth, AllocationPolicy>::intersection(const Qbasis<Symmetry, depth, AllocationPolicy>& other) const
{
    Qbasis<Symmetry, depth, AllocationPolicy> out;
    for(const auto& triple : this->data_) {
        if(auto it = other.cfind(std::get<0>(triple)); it != other.cend()) {
            auto dim = std::min(std::get<2>(triple).dim(), std::get<2>(*it).dim());
            out.push_back(std::get<0>(triple), dim);
        }
    }
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
Qbasis<Symmetry, 1, AllocationPolicy> Qbasis<Symmetry, depth, AllocationPolicy>::forgetHistory() const
{
    Qbasis<Symmetry, 1, AllocationPolicy> out;
    for(const auto& [q, num, plain] : data_) { out.push_back(q, plain.dim()); }
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
Qbasis<Symmetry, depth, AllocationPolicy> Qbasis<Symmetry, depth, AllocationPolicy>::conj() const
{
    assert(depth == 1);
    Qbasis<Symmetry, depth, AllocationPolicy> out;
    out.CONJ = not this->CONJ;
    for(const auto& [q, num, plain] : data_) {
        auto oldtrees = tree(q);
        assert(oldtrees.size() == 1);
        TreeVector<depth> trees;
        FusionTree<depth, Symmetry> tree;
        tree.q_uncoupled[0] = Symmetry::conj(q);
        tree.q_coupled = Symmetry::conj(q);
        tree.IS_DUAL[0] = not oldtrees[0].IS_DUAL[0];
        tree.dims[0] = oldtrees[0].dims[0];
        tree.dim = oldtrees[0].dim;
        trees.push_back(tree);
        out.push_back(Symmetry::conj(q), plain.dim(), trees);
    }
    out.sort();
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
Qbasis<Symmetry, depth + 1, AllocationPolicy> Qbasis<Symmetry, depth, AllocationPolicy>::combine(const Qbasis<Symmetry, 1, AllocationPolicy>& other,
                                                                                                 bool CONJ) const
{
    Qbasis<Symmetry, depth + 1, AllocationPolicy> out;
    for(const auto& elem1 : this->data_) {
        auto [q1, num1, plain1] = elem1;
        for(const auto& elem2 : other.data_) {
            auto [q2, num2, plain2] = elem2;
            if(CONJ) { q2 = Symmetry::conj(q2); }
            // auto plain = plain1.combine(plain2);
            auto qVec = Symmetry::reduceSilent(q1, q2);
            for(const auto& q : qVec) {
                for(const auto& thisTree : this->tree(q1))
                    for(const auto& otherTree : other.tree(q2)) {
                        auto totalTree = thisTree.enlarge(otherTree);
                        totalTree.q_coupled = q;
                        auto it = out.trees.find(q);
                        if(it == out.trees.end()) {
                            TreeVector<depth + 1> tree(1, totalTree);
                            out.trees.insert(std::make_pair(q, tree));
                        } else {
                            auto it_tree = std::find(it->second.begin(), it->second.end(), totalTree);
                            if(it_tree == it->second.end()) {
                                it->second.push_back(totalTree);
                            } else {
                                continue;
                            }
                        }
                    }
            }
        }
    }

    // sort the trees on the basis of Symmetry::compare()
    for(auto it = out.trees.begin(); it != out.trees.end(); it++) {
        std::vector<std::size_t> index_sort((it->second).size());
        std::iota(index_sort.begin(), index_sort.end(), 0ul);
        std::sort(index_sort.begin(), index_sort.end(), [&](std::size_t n1, std::size_t n2) {
            // return Symmetry::compare((it->second)[n1].source,(it->second)[n2].source);
            return (it->second)[n1] < (it->second)[n2];
        });
        TreeVector<depth + 1> tree = it->second;
        for(std::size_t i = 0; i < tree.size(); i++) { (it->second)[i] = tree[index_sort[i]]; }
    }

    // build up the new basis
    for(auto it = out.trees.begin(); it != out.trees.end(); it++) {
        size_t inner_dim = 0;
        for(const auto& i : it->second) { inner_dim += i.dim; }
        out.push_back(it->first, inner_dim, it->second);
    }
    out.sort();
    return out;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
std::string Qbasis<Symmetry, depth, AllocationPolicy>::info() const
{
    std::stringstream res;
    res << "Basis(" << Symmetry::name() << ", dim=" << dim();
    if constexpr(Symmetry::ANY_NON_ABELIAN) { res << "[" << fullDim() << "]"; }
    res << ")";
    if(IS_CONJ()) { res << "'"; }
    return res.str();
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
std::string Qbasis<Symmetry, depth, AllocationPolicy>::print() const
{
    std::string out = info();
    fmt::format_to(std::back_inserter(out), ": ");
    for(std::size_t i = 0; const auto& [q, num, plain] : data_) {
        if(i++ < data_.size() - 1) {
            fmt::format_to(std::back_inserter(out), "q={}[{}], ", Sym::format<Symmetry>(q), plain.dim());
        } else {
            fmt::format_to(std::back_inserter(out), "q={}[{}]", Sym::format<Symmetry>(q), plain.dim());
        }
    }
    return out;
    // tabulate::Table t;
    // using Row_t = std::vector<std::variant<std::string, const char*, tabulate::Table>>;
    // t.add_row(Row_t({"Q", "Dim(Q)", "num"}));
    // for(const auto& entry : data_) {
    //     auto [q_Phys, curr_num, plain] = entry;
    //     std::stringstream ss, tt, uu;
    //     ss << Sym::format<Symmetry>(q_Phys);
    //     tt << plain.dim();
    //     uu << curr_num;
    //     t.add_row(Row_t({ss.str(), tt.str(), uu.str()}));
    // }
    // t.format()
    //     .font_align(tabulate::FontAlign::center)
    //     .font_style({tabulate::FontStyle::bold})
    //     .border_top(" ")
    //     .border_bottom(" ")
    //     .border_left(" ")
    //     .border_right(" ")
    //     .corner(" ");
    // t[0].format().padding_top(1).padding_bottom(1).font_style({tabulate::FontStyle::underline}).font_background_color(tabulate::Color::red);
    // t[0][1].format().font_background_color(tabulate::Color::blue);
    // return t;
}

template <typename Symmetry, std::size_t depth, typename AllocationPolicy>
std::string Qbasis<Symmetry, depth, AllocationPolicy>::printTrees() const
{
    std::stringstream out;
    for(auto it = trees.begin(); it != trees.end(); it++) {
        out << it->second.size() << " Fusion trees for Q=" << Sym::format<Symmetry>(it->first) << std::endl;
        for(const auto& i : it->second) { out << i.draw() << std::endl; }
        out << std::endl << "******************************************************************************************" << std::endl;
        ;
    }
    return out.str();
}

} // namespace Xped

#if __has_include("Qbasis.gen.cpp")
#    include "Qbasis.gen.cpp"
#endif
