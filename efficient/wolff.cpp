// modules
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm> // for max
#include <fstream> // output to files
#include <sstream> // for string
#include <iomanip> // for string
#include <filesystem> // similar to os in python


using namespace std;
namespace fs = std::filesystem;

// defines

#ifndef L
#define L 100   // override at compile time with -DL=...
#endif
#define N (L*L)


// variables
const int J = 1;

int s[N];
double padd;
vector<int> clusterStack;

double T;
const double Tc= 2 * abs(J) / log(1 + sqrt(2));
int curM, curE;      // current E/M as running totals

// sampling settings
// one "sweep-equivalent" = ~N flipped spins, so one sample ~ one metropolis sweep
const int burnInSamples   = 200;  // production burn-in, in samples (at frozen kT)
const int pilotEqSweeps   = 300;  // pilot equilibration, in sweep-equivalents (N flips each)
const int pilotMeasSweeps = 100;  // pilot nbar measurement, in sweep-equivalents

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
// stage: 0 = pilot calibration, 1 = production (so they can never collide)
// L is included so different lattice sizes get independent streams
void seedRun(int stage, int t, int run) {
    seed_seq seq{baseSeed, (unsigned int)L, (unsigned int)stage,
                 (unsigned int)t, (unsigned int)run};
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

// sum of the 4 neighbours' current spins ,periodic
inline int nnSum(int i) {
    int r = i + 1; if (r >= N) r -= N;
    int l = i - 1; if (l < 0)  l += N;
    int u = i + L; if (u >= N) u -= N;
    int d = i - L; if (d < 0)  d += N;
    return s[r] + s[l] + s[u] + s[d];
}


// wolff step growing the cluster to flip
// curM and curE are updated incrementally
int wolffStep(){

    int i = randSite();
    int oldspin, newspin;
    int current, nn;
    int clusterSize =1;

    clusterStack.push_back(i);
    oldspin = s[i];
    newspin = -s[i];

    curE += 2*J*oldspin*nnSum(i);
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
            curE += 2*J*oldspin*nnSum(nn);
            s[nn] = newspin;
        }
        // x direction wraps first site to last site
        if ((nn=current-1)<0) nn += N;
        if (s[nn]==oldspin)
            if (drandom( )<padd) {
            clusterStack.push_back(nn);
            clusterSize++;
            curE += 2*J*oldspin*nnSum(nn);
            s[nn] = newspin;
        }
        // y direction wraps last site to first site
        if ((nn=current+L)>=N) nn -= N;
        if (s[nn]==oldspin)
            if (drandom( )<padd) {
            clusterStack.push_back(nn);
            clusterSize++;
            curE += 2*J*oldspin*nnSum(nn);
            s[nn] = newspin;
        }
        // y direction wraps first site to last site
        if ((nn=current-L)<0) nn += N;
        if (s[nn]==oldspin)
            if (drandom( )<padd) {
            clusterStack.push_back(nn);
            clusterSize++;
            curE += 2*J*oldspin*nnSum(nn);
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
      << "nsweeps "  << nsweeps  << "  (samples per run, ~N flipped spins each)\n"
      << "burnInSamples "   << burnInSamples   << "\n"
      << "pilotEqSweeps "   << pilotEqSweeps   << "\n"
      << "pilotMeasSweeps " << pilotMeasSweeps << "\n"
      << "baseSeed " << baseSeed << "\n"
      << "engine mt19937\n"
      << "seeding seed_seq{baseSeed, L, stage, t_index, run_index}  (stage 0 = pilot, 1 = production)\n"
      << "T_i = Tmin + (Tmax-Tmin)*i/(nT-1)\n"
      << "data columns: sample M E blockClusterSum  (M, E are totals, not per spin)\n"
      << "# per-temperature calibration: T kT nbar\n";
}

// pilot calibration for one temperature
// equilibrate and measure the mean cluster size nbar, both budgeted in flipped
// spins so every T gets similar effort, then freeze kT = round(N / nbar).
// the pilot's spin state is thrown away, production runs reseed and reinitialise.
void calibrate(int t, double& nbar, int& kT) {
    seedRun(0, t, 0);
    initialiseSpins();
    curM = magnetisation();
    curE = energy();

    // equilibrate, results discarded
    long long flips = 0;
    const long long eqBudget = (long long)pilotEqSweeps * N;
    while (flips < eqBudget) flips += wolffStep();

    // measure nbar = total flipped spins / total steps
    flips = 0;
    long long steps = 0;
    const long long measBudget = (long long)pilotMeasSweeps * N;
    while (flips < measBudget) {
        flips += wolffStep();
        steps++;
    }
    nbar = static_cast<double>(flips) / steps;
    kT = max(1, static_cast<int>(llround(N / nbar)));
}

// self-describing header for a data file
void writeHeader(ofstream& f, const char* kind, int run, int kT, double nbar,
                 int t, long long nrows) {
    f << "# " << kind << "\n"
      << "# L " << L << "\n"
      << "# N " << N << "\n"
      << "# T " << rounding(T, 5) << "\n"
      << "# run " << run << "\n"
      << "# seed_seq {" << baseSeed << ", " << L << ", 1, " << t << ", " << run << "}\n"
      << "# start " << ((T < Tc) ? "cold" : "hot") << "\n"
      << "# kT " << kT << "   (wolff steps per sample)\n"
      << "# nbar " << nbar << "\n"
      << "# burnInSamples " << burnInSamples << "\n"
      << "# nrows " << nrows << "\n"
      << "# columns: sample M E blockClusterSum\n";
}


int main() {
    const int nT=10; // number of temps in the range
    const int nsweeps = 10000; // samples per run, each ~N flipped spins (one metropolis-sweep equivalent)
    const int nRuns = 10;

    const double Tmin=1.8;
    const double Tmax=2.8;

    clusterStack.reserve(N);

    const string baseDir = "data/wolff_results/L" + to_string(L)+"/nt" 
        + rounding(nT, 0) +"/nsweeps" + to_string(nsweeps);
    writeMetadata(baseDir, nT, Tmin, Tmax, nRuns, nsweeps);
    
    
    // nT temps loop
    for (int t = 0; t < nT; t++) {
        T = Tmin + (Tmax - Tmin) * t / (nT - 1);
        padd = 1.0 - exp(-(2.0 * J) / T);

        // pilot: freeze kT (wolff steps per sample) for this temperature
        double nbar;
        int kT;
        calibrate(t, nbar, kT);

        cout << "T=" << T << " , "<< t+1<<"/"<<nT <<endl
            << "  nbar=" << nbar
            << "  kT=" << kT
            << "  samples/run=" << nsweeps << '\n';

        {
            ofstream m(baseDir + "/metadata.txt", ios::app);
            m << rounding(T, 5) << " " << kT << " " << nbar << "\n";
        }

        const string tDir = baseDir + "/T" + rounding(T, 5);
        fs::create_directories(tDir);

        for (int run=0; run < nRuns; run++){
            // new seed for RNG 
            seedRun(1, t, run);
            initialiseSpins();
    
            curM = magnetisation();
            curE = energy();

            // burn-in: burnInSamples samples at frozen kT, discarded
            // run 0 keeps its trace for the equilibration plot
            vector<int> bM, bE, bS;
            if (run == 0) {
                bM.reserve(burnInSamples);
                bE.reserve(burnInSamples);
                bS.reserve(burnInSamples);
            }
            for (int b = 0; b < burnInSamples; ++b) {
                int blockSum = 0;
                for (int k = 0; k < kT; ++k) blockSum += wolffStep();
                if (run == 0) {
                    bM.push_back(curM);
                    bE.push_back(curE);
                    bS.push_back(blockSum);
                }
            }
            if (run == 0) {
                ofstream bf(tDir + "/burnin_run000.dat");
                writeHeader(bf, "burn-in trace (not used for statistics)", run, kT, nbar, t, burnInSamples);
                for (int b = 0; b < burnInSamples; ++b) {
                    bf << b << " " << bM[b] << " " << bE[b] << " " << bS[b] << '\n';
                }
            }

            // production: one sample every kT wolff steps
            vector<int> mags, ens, sums;
            mags.reserve(nsweeps);
            ens.reserve(nsweeps);
            sums.reserve(nsweeps);

            const int printEvery = max(1, nsweeps / 100);
            for (int smp = 0; smp < nsweeps; ++smp) {
                int blockSum = 0;
                for (int k = 0; k < kT; ++k) blockSum += wolffStep();

                sums.push_back(blockSum);
                mags.push_back(curM);
                ens.push_back(curE);

                if ((smp + 1) % printEvery == 0 || smp + 1 == nsweeps) {
                    cout << "\r  run " << run + 1 << "/" << nRuns << "  "
                        << 100.0 * (smp + 1) / nsweeps
                        << "%" << "\033[K" << flush;
                }
            }
            cout<<endl;

            // bookkeeping check: running totals must match a full recalculation
            if (curE != energy() || curM != magnetisation()) {
                cerr << "Bookkeeping mismatch at T=" << T << " run " << run << endl;
                return 1;
            }

            ostringstream rn;
            rn << "run" << setw(3) << setfill('0') << run << ".dat";

            string filename = tDir + "/" + rn.str();
            ofstream file(filename);
            writeHeader(file, "production samples", run, kT, nbar, t, nsweeps);

            for (int smp = 0; smp < nsweeps; ++smp) {
                file << smp << " " << mags[smp] << " "
                    << ens[smp] << " " << sums[smp] << '\n';
            }
        }

        cout << endl;
    }
    return 0;
}