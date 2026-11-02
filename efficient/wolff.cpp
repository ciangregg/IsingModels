// modules
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream> // output to files
#include <sstream> // for string
#include <iomanip> // for string
#include <filesystem> // similar to os in python


using namespace std;
namespace fs = std::filesystem;

// defines

#define L 100
#define N (L*L)


// variables
const int J = 1;

int s[N];
double padd;
vector<int> clusterStack;

double T;
const double Tc= 2 * abs(J) / log(1 + sqrt(2));
int curM, curE;      // current E/M as running totals

// functions

// Pseudo-Random Number Generators


// mt19937 "Mersenne Twister" a standard PRNG
// same PRNG as np.random(), from what i can tell

const unsigned int baseSeed = 13;

mt19937 gen;
uniform_real_distribution<double> dist(0.0, 1.0); // for random numbers
uniform_int_distribution<int> siteDist(0, N - 1); // for radnom sites positions

double drandom() {
     return dist(gen); 
}

int randSite() { 
    return siteDist(gen); 
}

// spaces out seeds with close t & run
void seedRun(int t, int run) {
    seed_seq seq{baseSeed, (unsigned int)t, (unsigned int)run};
    gen.seed(seq);
}

// we could compare other PRNGs later


// initialising for each temp/run

// cold or hot start for our spins
void initialiseSpins(){
        if(T< Tc){
            // start with all spins up below T_c approx 2.269 -- cold start
            for (int i = 0; i < N; i++) {s[i] = 1;}
        }
        else{
            //random assignment of spins s[i] -- hot start
            for (int i = 0; i < N; i++) {                     
            s[i] = (drandom() < 0.5) ? 1 : -1;}
        }
}

// full calculation of E and M
int magnetisation(){
    int M=0;
    for (int i = 0; i < N; i++) {M += s[i];}
    return M;
}

int energy(){
    int E=0;
    for(int i=0; i<N; i++) {
        int right = i + 1;
        if(right>= N){right-=N;}

        int up = i + L;
        if(up>= N){up-=N;}

        E -= J*s[i]*(s[right]+s[up]);
    };
    return E;
}


// wolff step growing the cluster to flip
int wolffStep(){

    int i = randSite();
    int oldspin, newspin;
    int current, nn;
    int clusterSize =1;

    clusterStack.push_back(i);
    oldspin = s[i];
    newspin = -s[i];

    s[i]=newspin;
    while(!clusterStack.empty()){

        current = clusterStack.back(); // reads top element
        clusterStack.pop_back();  // removes the top element

        // x direction wraps last site to first site
        if ((nn=current+1)>=N){nn -= N;}
        if (s[nn]==oldspin)
            if (drandom( )<padd) {
            clusterStack.push_back(nn);
            clusterSize++;
            s[nn] = newspin;
        }
        // x direction wraps first site to last site
        if ((nn=current-1)<0) nn += N;
        if (s[nn]==oldspin)
            if (drandom( )<padd) {
            clusterStack.push_back(nn);
            clusterSize++;
            s[nn] = newspin;
        }
        // y direction wraps last site to first site
        if ((nn=current+L)>=N) nn -= N;
        if (s[nn]==oldspin)
            if (drandom( )<padd) {
            clusterStack.push_back(nn);
            clusterSize++;
            s[nn] = newspin;
        }
        // y direction wraps first site to last site
        if ((nn=current-L)<0) nn += N;
        if (s[nn]==oldspin)
            if (drandom( )<padd) {
            clusterStack.push_back(nn);
            clusterSize++;
            s[nn] = newspin;
        }
    }
    curM += 2 * newspin * clusterSize;

    return clusterSize;
}


// auxillary functions

// rounding doubles for neater filenames
string rounding(double x, int digits = 3) {
    ostringstream s;
    s << fixed << setprecision(digits) << x;
    return s.str();
}

// meta data
void writeMetadata(const string& dir, int nT, double Tmin, double Tmax,
                   int nRuns, int nsweeps) {
    fs::create_directories(dir);
    ofstream m(dir + "/metadata.txt");
    m << "L "        << L        << "\n"
      << "nT "       << nT       << "\n"
      << "Tmin "     << Tmin     << "\n"
      << "Tmax "     << Tmax     << "\n"
      << "nRuns "    << nRuns    << "\n"
      << "nsweeps "  << nsweeps  << "\n"
      << "baseSeed " << baseSeed << "\n"
      << "engine mt19937\n"
      << "seeding seed_seq{baseSeed, t_index, run_index}\n"
      << "T_i = Tmin + (Tmax-Tmin)*i/(nT-1)\n";
}

double pilotMeanClusterSize(int pilotUpdates) {
    long long totalClusterSites = 0;

    for (int step = 0; step < pilotUpdates; ++step) {
        totalClusterSites += wolffStep();
    }

    return static_cast<double>(totalClusterSites) / pilotUpdates;
}


int main() {
    const int nT=10; // number of temps in the range
    const int nsweeps = 10000;// ive confused sweeps and steps
    const int nRuns = 10;

    const double Tmin=1.8;
    const double Tmax=2.8;

    int  printingNo=nsweeps/100;

    vector<int> mags, ens, sizes;
    clusterStack.reserve(N);

    const string baseDir = "data/wolff_results/L" + to_string(L)+"/nt" 
        + rounding(nT, 0) +"/nsweeps" + to_string(nsweeps);
    writeMetadata(baseDir, nT, Tmin, Tmax, nRuns, nsweeps);
    
    
    // nT temps loop
    for (int t = 0; t < nT; t++) {
        T = Tmin + (Tmax - Tmin) * t / (nT - 1);
        padd = 1.0 - exp(-(2.0 * J) / T);

        // Pilot run for this temperature
        seedRun(t, nRuns);  // use a seed index not used by production runs
        initialiseSpins();
        curM = magnetisation();

        //runForEquivalentSweeps(1000);  // discard pilot burn-in
        double nbar = pilotMeanClusterSize(10000);

        long long nsweeps_eq = static_cast<long long>(
            llround(static_cast<double>(nsweeps) * N / nbar)
        );

        cout << "T=" << T << " , "<< t+1<<"/"<<nT <<endl
            << "  nbar=" << nbar
            << "  equiv Wolff steps=" << nsweeps_eq
            << "  target equivalent sweeps=" << nsweeps << '\n';


        for (int run=0; run < nRuns; run++){
            // new seed for RNG 
            seedRun(t, run);
            initialiseSpins();
    
            curM = magnetisation();

            vector<int> mags, ens, sizes;
            mags.reserve(nsweeps_eq);
            ens.reserve(nsweeps_eq);
            sizes.reserve(nsweeps_eq);

            for (long long step = 0; step < nsweeps_eq; ++step) {
                int clusterSize = wolffStep();

                sizes.push_back(clusterSize);
                mags.push_back(curM);
                ens.push_back(energy()); // this is a bottle neck next fix

                if ((step + 1) % 1000 == 0 || step + 1 == nsweeps_eq) {
                    cout << "\r  run " << run + 1 << "/" << nRuns << "  "
                        << 100.0 * (step + 1) / nsweeps_eq
                        << "%" << "\033[K" << flush;
                }
            }
            cout<<endl;

            ostringstream rn;
            rn << "run" << setw(3) << setfill('0') << run << ".dat";

            string filename = baseDir  + "/T" + rounding(T, 5) + "/" + rn.str();
            fs::create_directories(fs::path(filename).parent_path());
            ofstream file(filename);

            for (long long step = 0; step < nsweeps_eq; ++step) {
                file << step << " " << mags[step] << " "
                    << ens[step] << " " << sizes[step]<< " " << nsweeps_eq << '\n';
            }
        }

        cout << endl;
    }
    return 0;
}