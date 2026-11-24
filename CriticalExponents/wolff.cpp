// Wolff for fss
#include <iostream>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm> // for max
#include <fstream> // output to files
#include <sstream> // for string
#include <iomanip> // for string
#include <filesystem> // similar to os in python

#include <gsl/gsl_rng.h>

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

// Pseudo-Random Number Generators -GSL
const unsigned int baseSeed = 13;

gsl_rng* r = nullptr;

double drandom() {
    return gsl_rng_uniform(r);  // Uniform in [0, 1)
}

int randSite() {
    return gsl_rng_uniform_int(r, N); // Integer in [0, N)
}

// seeding
// GSL's RANLUX uses the lowest 31 bits of the seed

//   bits 0     stage  (0 = pilot, 1 = production)
//   bits 1-8   run    
//   bits 9-28  tKey   (round(T*1e5), so T < 10.48576)

// baseSeed and L are mixed in with an XOR, which keeps seeds unique within one L.
// tKey is the same 5dp value as the folder name, so a seed depends only on
// (L, T, stage, run), not on the temps file or the order temperatures are run in.

const int maxRuns = 256;
const unsigned int maxTKey = 1u << 20;

unsigned int tempKey(double temp) {
    return (unsigned int)llround(temp * 1e5);
}

// the scrambler
uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

unsigned long makeSeed(int stage, unsigned int tKey, int run) {
    // packing: bits 0,stage(0 = pilot, 1 = production), bits 1-8 run, bits 9-28 temps 
    const uint64_t packed = ((uint64_t)tKey << 9) | ((uint64_t)run << 1) | (uint64_t)stage;
    //  scramble (baseSeed, L), using splitmix64
    const uint64_t mix = splitmix64(((uint64_t)baseSeed << 32) | (uint64_t)L);
    //  combine with XOR, keep 31 bits
    return (unsigned long)((packed ^ mix) & 0x7FFFFFFFULL);
}

unsigned long seedRun(int stage, unsigned int tKey, int run) {
    unsigned long seed = makeSeed(stage, tKey, run);
    if (seed == 0) {   // GSL would silently turn 0 into 1
        cerr << "seed came out as 0 (stage " << stage << ", tKey " << tKey
             << ", run " << run << "); change baseSeed\n";
        exit(1);
    }
    gsl_rng_set(r, seed);
    return seed;
}

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


// auxiliary functions

// rounding doubles for neater filenames
string rounding(double x, int digits = 3) {
    ostringstream s;
    s << fixed << setprecision(digits) << x;
    return s.str();
}

// per-temperature metadata, written after all runs at this T have finished
// (so the runner can treat "meta.txt exists" as "this T is done")
void writeTMeta(const string& tDir, unsigned int tKey, int nRuns, int nsamples,
                int kT, double nbar) {
    ofstream m(tDir + "/meta.txt");
    m << "L "        << L        << "\n"
      << "N "        << N        << "\n"
      << "T "        << rounding(T, 5) << "\n"
      << "nRuns "    << nRuns    << "\n"
      << "nsamples " << nsamples << "  (samples per run, ~N flipped spins each)\n"
      << "kT "       << kT       << "  (wolff steps per sample)\n"
      << "nbar "     << nbar     << "\n"
      << "burnInSamples "   << burnInSamples   << "\n"
      << "pilotEqSweeps "   << pilotEqSweeps   << "\n"
      << "pilotMeasSweeps " << pilotMeasSweeps << "\n"
      << "start "    << ((T < Tc) ? "cold" : "hot") << "\n"
      << "engine gsl_" << gsl_rng_name(r) << "\n"
      << "baseSeed " << baseSeed << "\n"
      << "tKey "     << tKey << "  (round(T*1e5))\n"
      << "seeding (tKey<<9 | run<<1 | stage) XOR splitmix64(baseSeed<<32 | L), lowest 31 bits\n"
      << "pilotSeed " << makeSeed(0, tKey, 0) << "\n"
      << "columns sample M E blockClusterSum  (M, E are totals, not per spin)\n";
}

// pilot calibration for one temperature
// equilibrate and measure the mean cluster size nbar, both budgeted in flipped
// spins so every T gets similar effort, then freeze kT = round(N / nbar).
// the pilot's spin state is thrown away, production runs reseed and reinitialize.
void calibrate(unsigned int tKey, double& nbar, int& kT) {
    seedRun(0, tKey, 0);
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
                 unsigned long seed, long long nrows) {
    f << "# " << kind << "\n"
      << "# L " << L << "\n"
      << "# N " << N << "\n"
      << "# T " << rounding(T, 5) << "\n"
      << "# run " << run << "\n"
      << "# seed " << seed << "   (gsl_" << gsl_rng_name(r) << ")\n"
      << "# start " << ((T < Tc) ? "cold" : "hot") << "\n"
      << "# kT " << kT << "   (wolff steps per sample)\n"
      << "# nbar " << nbar << "\n"
      << "# burnInSamples " << burnInSamples << "\n"
      << "# nrows " << nrows << "\n"
      << "# columns: sample M E blockClusterSum\n";
}


// read temperatures, one per line, '#' comments and blank lines ignored
vector<double> readTemps(const string& fname) {
    ifstream f(fname);
    if (!f) {
        cerr << "cannot open temps file " << fname << endl;
        exit(1);
    }
    vector<double> temps;
    string line;
    while (getline(f, line)) {
        size_t c = line.find('#');
        if (c != string::npos) line = line.substr(0, c);
        istringstream ss(line);
        double x;
        if (ss >> x) {
            if (x <= 0 || tempKey(x) >= maxTKey) {
                cerr << "temperature " << x << " out of range (0, 10.48576)\n";
                exit(1);
            }
            // folder names and seeds use 5dp, so the temperature must be exactly 5dp
            if (fabs(x * 1e5 - llround(x * 1e5)) > 1e-6) {
                cerr << "temperature " << setprecision(10) << x
                     << " has more than 5 decimal places; round it to 5dp\n";
                exit(1);
            }
            temps.push_back(x);
        }
    }
    return temps;
}


int main(int argc, char* argv[]) {

    if (argc != 5) {
        cerr << "usage: " << argv[0] << " <temps_file> <nRuns> <nsamples> <outdir>\n";
        return 1;
    }
    const vector<double> temps = readTemps(argv[1]);
    const int nRuns    = stoi(argv[2]);
    const int nsamples = stoi(argv[3]); // samples per run, each ~N flipped spins (one metropolis-sweep equivalent)
    const string outDir = argv[4];

    if (temps.empty()) {
    cerr << "no temperatures found in " << argv[1] << endl;
    return 1;
    }
    if (nRuns < 1 || nRuns > maxRuns) {
        cerr << "nRuns must be between 1 and " << maxRuns << ", got " << nRuns << endl;
        return 1;
    }
    if (nsamples < 1) {
        cerr << "nsamples must be >= 1, got " << nsamples << endl;
        return 1;
    }

    //GSL
    r = gsl_rng_alloc(gsl_rng_ranlxd1); // double-precision RANLUX
    //r = gsl_rng_alloc(gsl_rng_mt19937);
    //r = gsl_rng_alloc(gsl_rng_ranlxs0); /// RANLUX
    //r = gsl_rng_alloc(gsl_rng_ranlxs1); //better RANLUX
    //r = gsl_rng_alloc(gsl_rng_ranlxs2); //best RANLUX

    clusterStack.reserve(N);

    const string baseDir = outDir + "/L" + to_string(L) + "/nsamples" + to_string(nsamples);
    const int nT = temps.size();

    // temps loop
    for (int t = 0; t < nT; t++) {
        T = temps[t];
        padd = 1.0 - exp(-(2.0 * J) / T);
        const unsigned int tKey = tempKey(T);

        // pilot: freeze kT (wolff steps per sample) for this temperature
        double nbar;
        int kT;
        calibrate(tKey, nbar, kT);

        cout << "L=" << L << "  T=" << T << " , " << t + 1 << "/" << nT << endl
            << "  nbar=" << nbar
            << "  kT=" << kT
            << "  samples/run=" << nsamples << '\n';

        // folder name built from tKey so it always matches the seed
        const string tDir = baseDir + "/T" + rounding(tKey / 1e5, 5);
        fs::create_directories(tDir);

        for (int run=0; run < nRuns; run++){
            // new seed for RNG 
            const unsigned long seed = seedRun(1, tKey, run);
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
                writeHeader(bf, "burn-in trace (not used for statistics)", run, kT, nbar, seed, burnInSamples);
                for (int b = 0; b < burnInSamples; ++b) {
                    bf << b << " " << bM[b] << " " << bE[b] << " " << bS[b] << '\n';
                }
            }

            // production: one sample every kT wolff steps
            vector<int> mags, ens, sums;
            mags.reserve(nsamples);
            ens.reserve(nsamples);
            sums.reserve(nsamples);

            const int printEvery = max(1, nsamples / 100);
            for (int smp = 0; smp < nsamples; ++smp) {
                int blockSum = 0;
                for (int k = 0; k < kT; ++k) blockSum += wolffStep();

                sums.push_back(blockSum);
                mags.push_back(curM);
                ens.push_back(curE);

                if ((smp + 1) % printEvery == 0 || smp + 1 == nsamples) {
                    cout << "\r  run " << run + 1 << "/" << nRuns << "  "
                        << 100.0 * (smp + 1) / nsamples
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
            writeHeader(file, "production samples", run, kT, nbar, seed, nsamples);

            for (int smp = 0; smp < nsamples; ++smp) {
                file << smp << " " << mags[smp] << " "
                    << ens[smp] << " " << sums[smp] << '\n';
            }
        }


        // written last: marks this temperature as complete
        writeTMeta(tDir, tKey, nRuns, nsamples, kT, nbar);
        cout << endl;
    }

    gsl_rng_free(r);
    return 0;
}