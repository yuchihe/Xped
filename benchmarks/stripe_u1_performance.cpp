// Fixed-work native Xped benchmark using the validated D=6, 1x2 stripe checkpoint.
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "nlohmann/json.hpp"
#include "stan/math/rev.hpp"
#include "Xped/Util/YasHelpers.hpp"
#include "Xped/Symmetry/U1.hpp"
#include "Xped/Symmetry/kind_dummies.hpp"
#include "Xped/PEPS/CTM.hpp"
#include "Xped/PEPS/LinearAlgebra.hpp"
#include "Xped/PEPS/Models/Heisenberg.hpp"
using Sym = Xped::Sym::U1<Xped::Sym::SpinU1>;
using Psi = Xped::iPEPS<double, Sym, false, false>;
using Env = Xped::CTM<double, Sym, 2, false, false>;
using ADPsi = Xped::iPEPS<double, Sym, false, true>;
constexpr Xped::Opts::CTMCheckpoint cp{true, true, true, true, true, true};
using ADEnv = Xped::CTM<double, Sym, 2, false, true, cp>;
using Ham = Xped::Hamiltonian<double, Sym>;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr std::size_t archive_flags = yas::file | yas::binary;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void require(bool b,const char* s) { if(!b)throw std::runtime_error(s); }
template<class E> auto energy(E& env,Ham& ham) {
    auto op=ham.asObservable();auto terms=Xped::avg(env,op);
    return (terms[0].sum()+terms[1].sum()+terms[2].sum()+terms[3].sum())/double(env.cell().uniqueSize());
}
template<class T> void save(const std::string& path,T& object) {
    yas::file_ostream out(path.c_str());yas::save<archive_flags>(out,object);
}
int main(int argc,char** argv) {
 try {
    require(argc>=6,"usage: benchmark input.psi boundary mode chi steps [repeats=3] [output.json]");
    std::cout<<std::unitbuf;
    auto psi=std::make_shared<Psi>();yas::load<archive_flags>(argv[1],*psi);
    require(psi->D==6&&psi->cell().Lx==1&&psi->cell().Ly==2&&psi->checkConsistency(),"Invalid stripe checkpoint");
    psi->updateAdags();
    const int chi=std::stoi(argv[4]),steps=std::stoi(argv[5]),repeats=argc>6?std::stoi(argv[6]):3;
    require(chi>0&&steps>0&&repeats>0,"Invalid benchmark dimensions");
    std::map<std::string,Xped::Param> params{{"Jxy",{1.}},{"Jz",{1.}},{"J2",{.65}},{"Bz",{0.}}};
    auto ham=std::make_unique<Xped::Heisenberg<Sym,double>>(params,psi->cell().pattern,
        Xped::Opts::Bond::H|Xped::Opts::Bond::V|Xped::Opts::Bond::D1|Xped::Opts::Bond::D2,Xped::Opts::DiscreteSym::None);
    const std::string mode=argv[3];
    Json result={{"mode",mode},{"D",6},{"chi",chi},{"steps",steps},{"repeats",repeats},{"parameters",psi->plainSize()}};
    if(mode=="prepare") {
        require(!std::filesystem::exists(argv[2]),"Refusing to overwrite boundary");
        auto t=Clock::now();Env env(psi,chi,Xped::Opts::CTM_INIT::FROM_A);env.init();
        for(int k=0;k<steps;++k){env.solve(1);std::cout<<Json{{"event","prepare_step"},{"step",k+1},{"seconds",seconds(t)}}.dump()<<'\n';}
        const double e=energy(env,*ham);require(std::isfinite(e),"Nonfinite preparation energy");
        save(argv[2],env);result["energy"]=e;result["seconds"]=seconds(t);
    } else {
        require(mode=="plain"||mode=="ad","Unknown mode");
        Env base;yas::load<archive_flags>(argv[2],base);base.set_A(psi);
        require(base.chi()==std::size_t(chi),"Boundary chi mismatch");
        result["samples"]=Json::array();
        for(int trial=-1;trial<repeats;++trial) {
            Json sample={{"trial",trial}};const auto total=Clock::now();
            if(mode=="plain") {
                Env env(base);env.set_A(psi);const auto t=Clock::now();env.solve(steps);
                sample["ctm_seconds"]=seconds(t);const auto te=Clock::now();double e=energy(env,*ham);
                require(std::isfinite(e),"Nonfinite plain energy");sample["energy"]=e;sample["energy_seconds"]=seconds(te);
            } else {
                stan::math::nested_rev_autodiff tape;
                auto adpsi=std::make_shared<ADPsi>(*psi);ADEnv env(base);env.set_A(adpsi);adpsi->updateAdags();
                const auto t=Clock::now();env.solve(steps);auto e=energy(env,*ham);
                sample["forward_seconds"]=seconds(t);sample["energy"]=e.val();
                const auto tb=Clock::now();stan::math::grad(e.vi_);sample["backward_seconds"]=seconds(tb);
                auto g=adpsi->graddata();require(std::isfinite(e.val()),"Nonfinite AD energy");
                for(double v:g)require(std::isfinite(v),"Nonfinite AD gradient");
                sample["gradient_norm"]=std::sqrt(std::inner_product(g.begin(),g.end(),g.begin(),0.));
                if(trial==0)result["gradient"]=g;
            }
            sample["total_seconds"]=seconds(total);
            std::cout<<Json{{"event","sample"},{"mode",mode},{"sample",sample}}.dump()<<'\n';
            if(trial>=0)result["samples"].push_back(sample);
        }
    }
    result["passed"]=true;if(argc>7)std::ofstream(argv[7])<<result.dump(2)<<'\n';
    std::cout<<Json{{"event","complete"},{"mode",mode},{"passed",true}}.dump()<<'\n';return 0;
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

