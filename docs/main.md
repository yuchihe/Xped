# Manual
\mainpage
Xped is c++ library for the definition and manipulation of symmetry preserving tensors for tensor network related algorithms.

# Theory

## Tensor

Multiple definitions exist for the term *tensor* (see e.g. [wikipedia](https://en.wikipedia.org/wiki/Tensor "Tensor")).
For the purpose of this library, a tensor \f$T\f$ is a multidimensional linear map between the tensor product of \f$d\f$ vector spaces \f$\mathbb{C}^{n_1}\otimes\cdots \otimes\mathbb{C}^{n_d}\f$ 
which form the domain \f$\mathcal{D}\f$ and a tensor product of \f$c\f$ vector spaces \f$\mathcal{\mathbb{C}^{m_1}\otimes\cdots \otimes\mathbb{C}^{m_c}}\f$  which form the codomain \f$\mathcal{C}\f$:
\f{align*}{
T:\mathbb{C}^{n_1}\otimes\cdots \otimes\mathbb{C}^{n_d} &\to \mathbb{C}^{m_1}\otimes\cdots \otimes\mathbb{C}^{m_c} \\
v_1\otimes\cdots\otimes v_d &\mapsto w_1\otimes\cdots\otimes w_c
\f}
The rank of the tensor \f$T\f$ is \f$r=d+c\f$. 
The dimension of the domain is \f$dim_{\mathcal{D}}=n_1\cdots n_d\f$ and of the codomain \f$dim_{\mathcal{C}}=m_1\cdots m_c\f$.
Since the map is multi-linear, its action is entirely defined by the action on the basis vectors.
This defines the components of the tensor with respect to given bases:
\f[
    j_1\otimes\cdots\otimes j_c =  \sum_{i_1,\dots,i_d}T_{i_1,\dots,i_d}^{j_1,\dots,j_c} i_1\otimes\cdots\otimes i_d
\f]

By a canonical isomorphism \f$\pi_\mathcal{D}\f$, one can define the composite vector space \f$\mathbb{C}^{dim_\mathcal{D}}\f$ for the domain of the map:
\f[
\mathbb{C}^{dim_\mathcal{D}}\cong\mathbb{C}^{n_1}\otimes\cdots \otimes\mathbb{C}^{n_d}
\f]
Analogously, one can define a composite space \f$\mathbb{C}^{dim_\mathcal{C}}\f$ for the codomain.
The tensor T is then an ordinary linear map:
\f[
T: \mathbb{C}^{dim_\mathcal{D}} \to \mathbb{C}^{dim_\mathcal{C}}
\f]
This map can be represented by a matrix which will be useful when storing the tensor in memory.

## Symmetric tensor

A symmetric tensor is a tensor which *respects* a given symmetry.
To formalize this, lets assume a symmetry described by a group \f$G\f$.
\f$G\f$ is represented on all individual vector spaces of the domain by \f$\mathcal{R}^i_G\f$ and codomain by \f$\mathcal{Q}^i_G\f$ of a tensor.
Then a tensor \f$T\f$ is symmetric if:
\f[
    T(v_1\otimes\cdots\otimes v_d) = \mathcal{Q}^1_G\otimes\cdots\otimes \mathcal{Q}^c_G\left( T(\mathcal{R}^1_G(v_1)\otimes\cdots\otimes \mathcal{R}^d_G(v_d))\right)
\f]
That means the action of the tensor stays the same when the symmetry acts on the individual spaces.
If the tensor is transformed with the isomorphisms \f$\pi_\mathcal{D}\f$ and \f$\pi_\mathcal{C}\f$ it becomes an ordinary linear map.
Then Schur's lemma ([wikipedia](https://en.wikipedia.org/wiki/Schur%27s_lemma "Schur's lemma")) implies that this map is a constant if the composite representation of the symmetry on domain and codomain is irreducible.
Generally, it will be reducible but may be decomposed into a direct sum of irreducible representations (irreps).
The underlying matrix becomes then block diagonal.

### Construction of isomorphism

The block diagonal form of the underlying matrix is only manifest if the isomorphism \f$\pi_\mathcal{D}\f$ maps to a basis which is the decomposed into the direct sum of irreps.
Hence, the crucial task is to find an appropriate isomorphism.
A straight-forward method for obtaining \f$\pi_\mathcal{D}\f$ is the use of fusion trees which perform the mapping by a sequential pairwise fusion of two vector spaces into their product state.
For a given binary fusion of two spaces, the Clebsch-Gordan expansion is used to find the decomposition of the product state into irreps.
Each intermediate space is then combined with next unpaired space until all spaces are combined to the final domain/codomain of the tensor.
It is important to store all the intermediate quantum numbers since for general symmetries, several different paths exist which lead to the same output.
An example fusion tree of four quantum numbers \f$q_1,\dots,q_4\f$ coupled to \f$q_f\f$ looks as follows:
\dot "Example of a canonical fusion tree"
digraph G {
    bgcolor="black"
    node [fontcolor=white color=white]
    edge [color=white style=filled]
    {rank=same; q1 [shape=doublecircle, label=q₁]; q2 [shape=doublecircle, label=q₂]; q3 [shape=doublecircle, label=q₃]; q4 [shape=doublecircle, label=q₄]}
    i12 [label=α, shape=invtriangle];
    i123 [label=β, shape=invtriangle];
    i1234 [label=γ, shape=invtriangle];
    i1 [shape=doublecircle, label=i₁];
    i2 [shape=doublecircle, label=i₂];
    qf [shape=doublecircle];
    q1->i12;
    q2->i12;
    i12->i1;
    i1->i123;
    q3->i123;
    i123->i2;
    i2->i1234;
    q4->i1234;
    i1234->qf;
}
\enddot
The triangular shapes correspond to the binary Clebsch-Gordan decomposition.
They carry another (Greek) label which is important for symmetries with outermultiplicity in their Clebsch-Gordan expansion (e.g. SU(3)).
For Abelian symmetries, the complete fusion tree is entirely determined by the incoming quantum numbers and the intermediate values do not have to be stored.

The fusion trees construct the isomorphisms \f$\pi_\mathcal{D}\f$ and \f$\pi_\mathcal{C}\f$. 
\note The isomorphisms \f$\pi_\mathcal{D}\f$ and \f$\pi_\mathcal{C}\f$ are not constructed explicitly but only implicitly by storing all fusion trees.

## Tensor operations

For tensor network related algorithms, several tensor operations are ubiquitous.
One needs to contract tensors together to obtain final results or needs to decompose a tensor with a singular value decomposition to truncate certain bonds in the network.
A fundamental operation is the permutation of the legs of an individual tensor. It is needed for other high-level operations.

### Permutations

A permutation is a reshuffling of the individual legs of the tensor.
This corresponds to a reordering of the individual vector spaces which form \f$\mathcal{D}\f$ and \f$\mathcal{C}\f$.
For nonsymmetric tensors, this can be achieved by choosing appropriate new isomorphisms \f$\pi_\mathcal{D}^\prime\f$ and \f$\pi_\mathcal{C}^\prime\f$ 
such that the permuted input is mapped to the correct state. This leads to a new view to the original data which can be left untouched.
However, the data layout becomes strided and noncontiguous.
For some operations, it is therefore more efficient to copy the underlying data so that the layout is again contiguous.

In the case of symmetric tensors, the situation is more complicated. 
A permutation requires a reordering of the incoming legs of the fusion trees.
If the incoming legs are shuffled, the resultant fusion tree is not canonical anymore and therefore not useful for further operations which expects a canonical fusion tree.
In order to get the reordered fusion tree in canonical form, one needs the recoupling symbols \f$F\f$ of the symmetry group \f$G\f$.
\dot "Recoupling symbols"
digraph G {
    bgcolor="black"
    node [fontcolor=white color=white]
    edge [color=white style=filled]
    {rank=same; q1 [shape=doublecircle, label=q₁]; q2 [shape=doublecircle, label=q₂]; q3 [shape=doublecircle, label=q₃]; q1p [shape=doublecircle, label=q₁]; q2p [shape=doublecircle, label=q₂]; q3p [shape=doublecircle, label=q₃]}
    {rank=same; i1 [shape=doublecircle, label=q₁₂]; eq [shape=none, label="=ΣF", fontsize="50pt"]; i2 [shape=doublecircle, label=q₂₃]}
    i12 [label=α, shape=invtriangle];
    i12_3 [label=β, shape=invtriangle];
    i23 [label=γ, shape=invtriangle];
    i1_23 [label=δ, shape=invtriangle];
    Q [shape=doublecircle];
    Qp [shape=doublecircle, label=Q];
    q1->i12;
    q2->i12;
    i12->i1;
    i1->i12_3;
    q3->i12_3;
    i12_3->Q;
    q2p->i23;
    q3p->i23;
    i23->i2;
    i2->i1_23;
    q1p->i1_23;
    i1_23->Qp;
    i1 -> eq [style=invis];
    eq -> i2 [style=invis];
    eq -> Qp [style=invis];
}
\enddot
The sum goes over all possible intermediate quantum numbers \f$q_{23}\f$ and in the case of outermultiplicity also over \f$\gamma\f$ and \f$\delta\f$.
The recoupling symbols define a unitary transformation between the composite spaces of different fusion orders.
They are entirely determined by the Clebsch Gordan coefficients.

To obtain the reordered fusion tree in canonical form, one needs a second basis operation which exchanges two incoming quantum numbers on the same node.
This operations involves the swap symbol for the group.

With these two operations, one can implement arbitrary permutations on a single fusion tree.
A tensor consists of two fusion trees, one for the domain and one for the codomain.
For permutations which shuffle across the domain and codomain, a third basic operation is needed.
This involves the bending of a line pointing inwards to pointing outwards and is quantified by the turn symbol of the group.

The permutation of the legs of a symmetric tensor is then performed by first permuting the fusion tree pair (consisting of domain tree and codomain tree) 
and second by assigning the old data to the new tensor with careful attention to the recoupling-, swap- and turn symbols.

### Contractions

A tensor contraction boils down to the composition of the associated maps.
I.e., if \f$T_1: \mathcal{D}\to\mathcal{I}\f$ and \f$T_2: \mathcal{I}\to\mathcal{C}\f$ than the contraction \f$T\f$ of \f$T_1\f$ and \f$T_2\f$ over all the elementary spaces in \f$\mathcal{I}\f$ is the composition:
\f[
    T = T_2 \circ T_1
\f]
The composition of tensors corresponds to the multiplication of the representing matrices which arise after the application of the isomorphisms \f$\pi_\mathcal{D}\f$ and \f$\pi_\mathcal{C}\f$.

In the generic case, the indices of a tensor which should be contracted with indices of another tensor are located in its codomain.
Therefore one has to permute the indices of both tensors so that an ordinary composition performs the contraction.
After the composition, another permute might bring the indices in the desired order for the final result.

### Decompositions

A tensor decomposition can be seen as the inverse operation to a contraction.
Because of that it can be handled similarly.
An often encountered decomposition is the singular value decomposition ([svd](https://en.wikipedia.org/wiki/Singular_value_decomposition "singular value decomposition")).
It can be applied to any linear map.
Hence, the tensor is again interpreted as an ordinary linear map with the help of the isomorphisms \f$\pi_\mathcal{D}\f$ and \f$\pi_\mathcal{C}\f$.
Then, the svd reads:
\f[
    T = U \Sigma V^\dagger
\f]
\f$U\f$ has the original domain of the tensor and \f$V\f$ has the original codomain, i.e. the cut is taken through the domain and codomain.

For a general partition of indices, one can apply a permute operation to get an appropriate domain and codomain.

# Implementation

## Quick tour

To get a feeling of how the library works, lets look at some simple examples for the core part of the library.
All entities of the library are defined in `namespace Xped`.
\include quickstart.cpp

## Symmetries

Each supported symmetry has its own class which supplies all methods related to the symmetry.
Currently implemented are the following groups:
* \f$Z_n\f$ \f$\rightarrow\f$ `template<std::size_t N, Xped::Sym::Kind> struct Xped::Sym::Zn;`
* \f$\text{U}(1)\f$ \f$\rightarrow\f$ `template<Xped::Sym::Kind> struct Xped::Sym::U1;`
* \f$\text{SU}(2)\f$ \f$\rightarrow\f$ `template<Xped::Sym::Kind> struct Xped::Sym::SU2;`
* trivial \f$\rightarrow\f$ `struct Xped::Sym::U0;`

Partial support exists also for general \f$\text{SU}(N)\f$ symmetry.

All symmetry classes have a [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern "CRTP") base class `template<typename Derived> struct Xped::Sym::SymBase;` to avoid duplicated code.
The individual symmetry classes are stateless and methods are static.
Some basic objects which are implemented in the symmetry classes are:
* `using qType = ...` the type for the quantum numbers of the symmetry
* `static std::vector<qType> basis_combine(qType ql, qType qr)` returns the decomposition of the direct product of two irreps into irreps.
* `static constexpr qType qvacuum();` returns the quantum number of the trivial representation.

To see the full list of required functions check `Symmetry/SU2.hpp`

All symmetry classes take `Xped::Sym::Kind` as a template parameter. 
It describes to which particles the symmetry relates to.
Possible values are:
* `Xped::Kind::Spin`
* `Xped::Kind::Boson`
* `Xped::Kind::Fermion`

The difference between the first two is only when formatting related quantum numbers because spin quantum numbers can be half-integers but particle quantum numbers are only integers.
The latter however, makes a crucial difference. If a symmetry is set up with `Xped::Kind::Fermion`, the fermionic commutation rules and respective signs are taken care off automatically for tensor operations.
This is controlled by the methods
```{.cpp}
static Scalar coeff_swap(qType ql, qType qr);
static Scalar coeff_twist(qType q);
```
The first methods adds a minus sign if two odd fermion-parity states are swapped.
The second includes a sign if a odd parity state is twisted.

## Fusion trees

The struct `FusionTree` is a simple static storage of all relevant data for a single fusion path of `Rank` quantum numbers of a given `Symmetry`:
```{.cpp}
template<std::size_t Rank, typename Symmetry>
struct FusionTree;
```
It has the following members:
```{.cpp}
std::array<qType, Rank> q_uncoupled{};
qType q_coupled{};
std::size_t dim{};
std::array<size_t, Rank> dims{};
std::array<qType, util::inter_dim(Rank)> q_intermediates{};
std::array<size_t, util::mult_dim(Rank)> multiplicities = std::array<size_t, util::mult_dim(Rank)>(); // only for non-Abelian symmetries with outermultiplicity.
std::array<bool, Rank> IS_DUAL{};
```
\warning This class does not check whether all the quantum numbers do fit together but it expects that it is constructed correctly.

To display a `FusionTree`, there is the handy function `Xped::FusionTree::draw()`.
It generates an ascii drawing of the tree. Here is an example for a rank=4 tree for an SU(2) symmetry:
```
 2       0        1/2       1/2
   \     /        /        /
    \   /        /        /
      μ         /        /
       \       /        /
        \ 2   /        /
         \   /        /
           μ         /
            \       /
             \ 3/2 /
              \   /
                μ
                |
                |
                1
```

### Fusion tree manipulations 

Fusion tree manipulations build the basis for tensor operations.
The fundamental operation is the permutation of two adjacent incoming quantum numbers.
Unless it involves the first two incoming quantum numbers which enters to the same binary fusion, this involves recoupling operations.
Here, especially the \f$6j\f$-symbol of the symmetry group enters.
At the end, a fusion tree can be written as a weighted sum over fusion trees with two adjacent quantum numbers exchanged.
The respective method is `Xped::FusionTree::swap`:
```{.cpp}
/*Swaps quantum numbers pos and pos+1*/
std::unordered_map<FusionTree<Rank, Symmetry>, typename Symmetry::Scalar> swap(const std::size_t& pos) const;
```
With the exchange of adjacent quantum numbers, one can implement arbitrary permutations of the incoming quantum numbers by decomposing the permutation into a chain of adjacent swaps.
This is performed with `Xped::util::Permutation::decompose()`.
The high-level function which performs the general permutation is `Xped::FusionTree::permute()`.

## Hilbert bases

To describe a Hilbert basis, the class `Qbasis` is used:
```{.cpp}
template<typename Symmetry, std::size_t depth, typename AllocationPolicy = HeapPolicy>
class Xped::Qbasis;
```
This class stores a basis which is properly sorted with respect to a given `Symmetry`, i.e. it contains several multiplets of the symmetry.
Furthermore, the basis can be a composite basis of several individual bases. This is controlled via the template parameter `depth`.
To handle the composition, `Qbasis` uses `FusionTree` to remember the exact fusion process.
Generally, bases are constructed with `depth=1` and composite bases are obtained using the method `Xped::Qbasis::combine()`.

The most important member of `Qbasis` is:
```{.cpp}
std::vector<std::tuple<qType, std::size_t, Xped::Basis>> data_;
```
The actual type is more complicated because of `AllocationPolicy` but this does not matter here.
Each entry of `data_` corresponds to a quantum number of type `qType` (first entry in the `tuple`).
For each quantum number, the global number of the first plain state corresponds to the second element in the `tuple`.
The third element of the `tuple` belongs to the plain basis states of this quantum number. The plain basis state are described by the simple `class Xped::Basis`.

A few lines of code illustrate the use of the class:
\include s_qbasis.cpp

One of the key methods of `Qbasis` are `Xped::Qbasis::leftOffset()` and `Xped::Qbasis::rightOffset()`.
If several bases are combined into a composite basis, each basis state in the composite basis belongs to a specific fusion.
A rigorous order of all the fused states with same quantum number is fundamental to construct the isomorphism \f$\pi_\mathcal{D}\f$.
The exact signature of `leftOffset()` is:
```{.cpp}
std::size_t leftOffset(const FusionTree<Symmetry, depth>& tree) const;
```
It takes a `FusionTree` as a parameter and returns the number of basis states which come *before* this tree.
These methods are fundamental for getting tensor views from the large block diagonal matrix which is stored in memory.
See the implementation of `Xped::Tensor::subBlock()`:
\snippet Core/Tensor.cpp Showcase use of leftOffset

## Tensor class

The tensor class is the central object of the library:

```{.cpp}
template<typename Scalar, std::size_t Rank, std::size_t CoRank, typename Symmetry, bool AD=false, typename AllocationPolicy=HeapPolicy>
class Tensor;
```
It takes 6 template parameters but the last two are optional.
The first four describe the underlying `Scalar` type, the rank and corank of the tensor (\f$d\f$ and \f$c\f$) and the symmetry under which the tensor is invariant.
The fifth parameter decides whether the tensor can be used for automatic differentiation (AD).
It is a `bool` parameter and both choices (`true` and `false`) corresponds to different explicit specializations of the general class template.
For the AD part see [below](#automatic-differentiation).
The last parameter controls the allocation procedure.

The tensor class has only a single member variable:

```{.cpp}
Storage storage_;
```

The used storage is currently determined at configuration time by a cmake parameter.
In the future, this can become an additional template parameter.

The most important methods are:
* `Xped::Tensor::permute();`
* `Xped::Tensor::contract();`
* `Xped::Tensor::tSVD();`

### Storage

For an efficient storage of the tensor data in the case of a symmetry constrain, the tensor is seen as an ordinary linear map between two vector spaces.
This procedure was described in the Theory section above.
In the case of a symmetry, the representing matrix of the linear map is block-diagonal.
Therefore, the most compact storage is obtained when storing only the nonzero block-matrices of the large block-diagonal matrix.
Each block is associated to a quantum number of the symmetry and contains all elements of the symmetry which can be fused to this quantum number.

Two specific storage implementations are available:
1. `VecOfMat` see Core/storage/StorageType_vecofmat.hpp
2. `Contiguous` see Core/storage/StorageType_contiguous.hpp

The first option stores a `std::vector` of matrices to describe the block-diagonal matrix.
For this option the entire tensor is not contiguous in memory.

The second option stores a large contiguous buffer (`std::vector<Scalar>`). 
When a specific block of the block-diagonal matrix is requested, a view into this large buffer is returned which corresponds to the requested block.

The `Storage` class must have two specific constructors.
```{.cpp}
    StorageType(const std::array<Xped::Qbasis<Symmetry, 1, AllocationPolicy>, Rank> basis_domain,
                const std::array<Xped::Qbasis<Symmetry, 1, AllocationPolicy>, CoRank> basis_codomain,
                const mpi::XpedWorld& world)
```
```{.cpp}
    StorageType(const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, Rank> basis_domain,
                const std::array<Qbasis<Symmetry, 1, AllocationPolicy>, CoRank> basis_codomain,
                const Scalar* data,
                std::size_t size,
                const mpi::XpedWorld& world)
```

Both take the individual bases of the domain and codomain as an `std::array` and the world to which the storage is associated in the case of distributed parallelism.
The first constructor does not allocate actual memory but only sets the meta data of the storage.
The second constructor allocates the needed storage and copies the contiguous data `data` into the tensor.
If `size` does not match the total elements in the tensor, an assertion is raised.

Additionally, the storage implementations need to include the following public methods.
* `const MatrixType& block(std::size_t i) const;` returns a const reference to the ith block of the block-diagonal matrix
* `MatrixType& block(std::size_t i);` returns a nonconst reference to the ith block of the block-diagonal matrix
* `const MatrixType& block(qType q) const;` returns a const reference to the block belonging to the fused quantum number `q`.
* `MatrixType& block(qType q);` returns a nonconst reference to the block belonging to the fused quantum number `q`.
* `std::size_t plainSize() const;` returns the total number of scalars in the tensor summed over all blocks.
* `const mpi::XpedWorld& world() const` returns the mpi world, the data is living in. This is only relevant for distributed tensors.

For the complete interface check out Core/storage/StorageType_contiguous.hpp.

### Automatic differentiation

The `Tensor` class supports build in automatic differentiation (AD), i.e. for algorithms build on top of the `Tensor` class, the derivative can be computed automatically.
To declare a tensor with AD support, one needs to set the respective template argument to `true`.
The source code for the AD support for `Tensor` can be found in `include/Xped/AD/ADTensor.hpp` (and not in `Tensor.hpp` itself).
It is an explicit specialization and of the `Tensor` class template. 
The underlying reverse-mode tape engine is self-contained and ships with Xped (`include/selfad`); it implements the small subset of the [stan/math](https://github.com/stan-dev/math "ad library") reverse-mode API that Xped uses (same `stan::math` namespace and include paths) without any external dependency.
The original external stan-math backend can still be selected with `-DXPED_USE_EXTERNAL_STAN=ON` (plus the `external-stan` vcpkg feature), which additionally requires stan-math, TBB and SUNDIALS.

With the Eigen matrix backend, CTM enlarged-corner contractions loop over blocks of a free boundary chi leg (up to eight multiplicity states per symmetry sector).
The sliced leg enters the first contraction and remains free throughout the intermediate tensors; the complete output corner is still stored for the projector decomposition.
The reverse pass recomputes and differentiates one block at a time on a nested tape, accumulating shared-input gradients and scattering the sliced-edge gradient.
This also applies when corner checkpointing is enabled, so AD retains no collection of per-block intermediates on the outer tape.

Let's look at an example usage of the AD functionality:
\include ad.cpp

-------------------------------------------------------------------------------

# Backends

The Xped library is designed to be a high-level library for symmetric tensors. 
All operations for plain nonsymmetric tensors are reimplemented. 
Instead, one can choose between different backends that deliver the functionality for plain tensors (or matrices).
It is also possible to add a backend by implementing the required interface.

Currently, three backends are available. The Eigen backend is tested most thoroughly. 

## Eigen

[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) is a famous c++ library for linear algebra.
It is designed with an elegant API and highly optimized matrix operations.
Additionally, it has a tensor module which is the basis for the TensorFlow framework by google.
This part supports arbitrary ranked tensors and a wealth of operations including contractions and reductions.

The Eigen library also supports the dispatch of fundamental matrix operations to external BLAS and LAPACK implementations.
This feature can be used within Xped at configuration time using the `XPED_USE_BLAS` and similar options.
We recommend Intel MKL for BLAS and LAPACK; enable it with `-DXPED_USE_MKL=ON`.
For SVD, the recommended driver is the divide-and-conquer `gesdd` (`dgesdd` for real double precision), rather than `gesvd`.
Xped uses Eigen's `BDCSVD` by default, which dispatches to `gesdd` when LAPACKE is enabled (including MKL builds); without LAPACKE it uses Eigen's native implementation.
The legacy `JacobiSVD`/`gesvd` path remains available by defining `XPED_DONT_USE_BDCSVD` at compile time.

## Cyclops tensor framework (ctf)

[ctf](https://solomon2.web.engr.illinois.edu/ctf/) is a tensor framework for distributed parallelism. 
This allows the computations to spread over several different compute cores and allows to use the hardware more efficient.
It has an overhead for small tensors but allows to scale the compute power with very large tensors.

\note The ctf backend is most useful for tensors without symmetry or with tensors with small internal symmetries.

## array

\todo Add documentation for array backend.

\warning The array backend needs some adaptions to be consistent with the code base.

# Algorithms

On top of the core library, Xped also provides high-level tensor network algorithms.

## Matrix product states (MPS)

MPSs represent a one-dimensional tensor network tailored for strongly interacting one-dimensional physical lattice systems.
For detailed information, check the following references:
* https://link.springer.com/article/10.1007/BF02099178
* https://arxiv.org/abs/1008.3477
* https://arxiv.org/abs/2011.12127

The MPS code is located in the directory `Xped/MPS` and currently provides a `class MPS` and basic operations for them.
An implementation of matrix product operators (MPO)s and algorithms like the density matrix renormalization-group (DMRG) or the time-dependent variational principle (TDVP)
are future development goals.

A small example demonstrates the current capability:
\include s_mps.cpp

## Projected entangled pair states (PEPS)

PEPS is a generalization of MPS for two-dimensional systems.
For detailed information, check the following references:
* https://arxiv.org/abs/cond-mat/0407066
* https://arxiv.org/abs/2011.12127

The PEPS code of Xped is located in `Xped/PEPS` and has the following capabilities:
* CTMRG algorithm for arbitrary unit cells with custom pattern (AB/BA)
* Imaginary time evolution using the simple update method with nearest and next-nearest Hamiltonian terms
* Direct energy-minimization using automatic differentiation

### Exact energy slicing

For the Eigen backend, `avg` and `CTMSolver` use exact energy slicing for real double-precision U0 two-site observables with `TRank=2` and `ALL_OUT_LEGS=false`.
The explicit entry point `avg_sliced(env, op, block_size=8)` selects the maximum number of boundary rows processed together; the width must be positive.
The evaluator supports site-dependent physical and virtual dimensions and the unit-cell pattern carried by the supplied tensors. Other scalar types, symmetries, and CTM layouts retain the existing observable path.

Each interaction is decomposed by a complete operator Schmidt SVD, with no rank truncation.
The calculation loops over the resulting operator channels, sums their ket/bra physical indices before forming large boundary intermediates, and processes blocks of a free boundary chi leg.
Horizontal and vertical nearest-neighbor terms retain their original 2-by-1 and 1-by-2 contraction geometries; the two diagonal terms retain their 2-by-2 geometry.
Each bond uses the normalization of its own geometry and evaluates the full supplied operator at finite chi, agreeing with the untruncated contraction network up to floating-point roundoff.

The historical diagonal observable path used `tSVD` with multiplet preservation enabled, which can discard trailing operator components even with an unlimited rank and zero threshold.
The sliced path retains every nonzero operator channel, so a general diagonal operator can give a different result from that historical calculation.
For a full reference using the original contraction network, first compute the RDMs and call `avg_unsliced(env, op, false)`; its default `preserve_operator_multiplets=true` retains the historical behavior used by fallback paths.

The reverse pass replays one operator contribution at a time on the native nested AD tape, while corner workspaces and their backward passes are processed in boundary blocks.
It accumulates gradients into every shared site and boundary input, including both occurrences of the site tensor in the double layer, and differentiates the normalization denominator.
This keeps per-contribution intermediates off the outer tape. Explicit output corners and their adjoints remain stored: slicing reduces temporary workspace, not every part of the memory footprint.
For example, with double precision, D=6 and chi=64, an individual enlarged-corner workspace falls from 40.5 MiB to 5.0625 MiB with eight boundary rows; the complete enlarged corner remains 40.5 MiB.

An earlier diagonal-only reflection benchmark for the spin-1/2 Heisenberg interaction measured the following results at D=6, chi=64, with 32 CTMRG updates from a fixed imported boundary and two BLAS/OpenMP threads:

| Boundary topology | Whole-job peak RSS, baseline -> sliced | Absolute energy difference | Relative L2 tensor-gradient difference |
| --- | --- | --- | --- |
| Two | 20.8978 -> 3.9793 GiB | 1.30e-14 | 2.16e-12 |
| Four | 20.9033 -> 4.0008 GiB | 1.41e-13 | 2.47e-11 |

That benchmark's exact evaluator used the complete three-product Heisenberg interaction; its nonzero operator channels are unaffected by the historical cutoff described above.
These measurements combine exact operator decomposition, contraction scheduling, and checkpoint replay; they do not isolate the boundary loop.
They are single shared-host samples of the earlier NNN implementation, whose NN calculation was unchanged, and do not establish a performance gain for the NN extension.
The full tensor gradients were compared with the original calculation; these high-chi runs did not perform finite differences.
Their derivatives are through the 32 finite CTMRG updates, rather than an implicit fixed-point derivative.

For PEPS simulations, the configuration is parsed as a toml file.
The toml file has sections for CTM, imaginary time evolution and nonlinear optimization.
The used model is specified in the section `[model]` and needs to be present in all simulations:
Here is an example for the Hubbard model:
```
[model]
name                    = "Hubbard"
# specifies on which bonds the two-site gates of the Hamiltonian are active. In the case of Hubbard model, this is the hopping term.
# V: vertical, H: horizontal, D1: diagonal top-left -> bottom-down, D2: diagonal bottom-left -> top-right
bonds                   = ["V", "H", "D1"] # This leads to a triangular lattice
[model.params]
U                       = 8.0
t                       = 1.0
tprime                  = 1.0
mu                      = 0.0
```

The unit cell and PEPS bond dimension are specified in the section `[ipeps]`:
```
[ipeps]
D                       = 2
# The cell can have a nontrivial pattern if numbers appear several times
pattern                    = [[1,2],
                              [2,1]] # 2x2 unit cell with pattern
#
# Alternative: specify cell as [Lx, Ly] without pattern
# cell                     = [3, 6]
#
# Charges can be used to force the total filling or total magnetization
# This is only possible for Abelian symmetries
# The charges must be consistent with the pattern of the cell!
charges                 = [[[+1], [-1]],
                           [[-1], [+1]]] # This charge pattern forces a Neel state with zero total magnetization
```

### CTM

The CTM procedure can be configured using toml configuration section `[ctm]`
```
[ctm]
chi                     = 30
max_presteps            = 30 # Maximum amount of steps without AD backtracking
track_steps             = 4 # Fixed number of steps for AD backtracking after the presteps are run. (Only used for gradient based optimization)
tol_E                   = 1e-10
reinit_env_tol          = 1e-3
init                    = "FROM_A" # Controls how the C and T-tensors are initialized
verbosity               = "PER_ITERATION"
```

### Simple update

Imaginary time evolution can be configured through the toml section `[Imag]`.
```
[Imag]
update                  = "SIMPLE" # Currently this is the only supported update. After implemented, this can be set to e.g. FULL for full update
tol                     = 0.1
resume                  = false
dts                     = [0.1, 0.05, 0.02, 0.01]
t_steps                 = [60, 60, 60, 60]
Ds                      = [2, 3, 4, 5]
chis                    = [[20, 30], [30], [40], [50]]
load                    = "path/to/previous/state"
load_format             = "NATIVE" # NATIVE: State saved within Xped, MATLAB: State saved from matlab simulation
working_directory       = "<wd>"
obs_directory           = "obs"
logging_directory       = "log"
seed                    = 10
id                      = 1
verbosity               = "PER_ITERATION"
```

### Energy minimization

Simulations for direct energy minimization using nonlinear optimization procedures are configured through the toml section `[optim]`.

```
[optim]
algorithm               = "L_BFGS" # Could also be CONJUGATE_GRADIENT or what a custom optimization library supports
linesearch              = "WOLFE"
bfgs_scaling            = false
grad_tol                = 1.0e-4
cost_tol                = 1.0e-8
step_tol                = 1.0e-10
resume                  = false
load                    = "/path/to/other/state"
load_format             = "NATIVE" # NATIVE: State saved within Xped, MATLAB: State saved from matlab simulation
max_steps               = 50
log_format              = ".log" # can also be set to .h5 for hdf5 log output
working_directory       = "<wd>"
obs_directory           = "obs"
logging_directory       = "log"
seed                    = 1
id                      = 1
verbosity               = "PER_ITERATION"
```
