// modules
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream> // output
#include <sstream> // for string
#include <iomanip> // for string
#include <filesystem> // similar to os in python


using namespace std;
namespace fs = std::filesystem;

// defines

#define L 100
#define N (L*L)
#define XNN 1
#define YNN L

// variables
const int J = 1;

int s[N];
double prob[5];
double T;
const double Tc= 2 * abs(J) / log(1 + sqrt(2));
int curM, curE;      // running totals


// Pseudo-Random Number Generator
// Randomn numbers are harder in C than python

//first attempt
/*double drandom(){
    double randomNum100 = rand() % 101;
    return randomNum100/100;
}*/

// mt19937 "Mersenne Twister" a pseduo rand number gen with a seed 
// same PRNG as np.random()
const unsigned int baseSeed = 13;

mt19937 gen;
uniform_real_distribution<double> dist(0.0, 1.0); // for random no
uniform_int_distribution<int> siteDist(0, N - 1); // for radnom sites

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

// comare other PRNGs



// Functions 

// initialising for each temp/run
// calculate the exp energy difference for each Temp, to avoid calculating exp each sweep 
void initialiseProb() {
    for (int i = 2; i < 5; i += 2) {
        prob[i] = exp(-(2.0 * i )/ T);
    }
}

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

// core metropolis sweep algorithm

void sweep() {
    for (int k = 0; k < N; k++) {
        int i = randSite();
        int nn, sum;

        if ((nn = i + XNN) >= N) nn -= N;
        sum = s[nn];
        if ((nn = i - XNN) < 0) nn += N;
        sum += s[nn];
        if ((nn = i + YNN) >= N) nn -= N;
        sum += s[nn];
        if ((nn = i - YNN) < 0) nn += N;
        sum += s[nn];

        int delta = sum * s[i];

         if (delta <= 0 || drandom() < prob[delta]) {
            curM -= 2 * s[i];     // M changes by -2s
            curE += 2 * delta;    // dE = 2*J*s_i*sum 
            s[i] = -s[i];
        }
    }
}

// full calculation of E and M
int magnetisation(){
    int M=0;
    for (int i = 0; i < N; i++) {
        M += s[i];
    }
    return M;
}

int energy(){
    int E=0;
    for(int i=0; i<N; i++) {
        int right = i + XNN;
        if(right>= N){right-=N;}

        int up = i + YNN;
        if(up>= N){up-=N;}

        E -= J*s[i]*(s[right]+s[up]);
    };
    return E;
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



int main() {
    const int nT=20; // number of temps in the range
    const int nsweeps = 10000;
    const int nRuns = 10;

    const double Tmin=1.8;
    const double Tmax=2.8;

    int  printingNo=nsweeps/100;

    vector<int> mags(nsweeps), ens(nsweeps);

    const string baseDir = "data/met_results/L" + to_string(L) +
                        "/nsweeps" + to_string(nsweeps);
    writeMetadata(baseDir, nT, Tmin, Tmax, nRuns, nsweeps);
    
    // nT temps loop
    for (int t = 0; t < nT; t++) {
        T = Tmin + (Tmax - Tmin) * t / (nT - 1);
        initialiseProb();
        cout << "T=" << T << " , "<< t+1<<"/"<<nT <<endl;
        cout << "prob[2]=" << prob[2] << "  prob[4]=" << prob[4] << endl;

        for (int run=0; run < nRuns; run++){
            // new seed for RNG 
            seedRun(t, run);  
            initialiseSpins();

            curM = magnetisation();  
            curE = energy(); 

            for(int step = 0; step < nsweeps; step++) {
                sweep();
                mags[step] = curM;
                ens[step]  = curE;
                if ((step + 1) % printingNo == 0) {
                cout << "\r  run " << run + 1 << "/" << nRuns << "  "
                    << 100 * (step + 1) / nsweeps << "%" << "\033[K" << flush;
                }
            }
            cout<<endl;

            ostringstream rn;
            rn << "run" << setw(3) << setfill('0') << run << ".dat";

            string filename = baseDir + "/T" + rounding(T, 5) + "/" + rn.str();
            fs::create_directories(fs::path(filename).parent_path());
            ofstream file(filename);
            
            for (int step = 0; step < nsweeps; step++)
                file << step << " " << mags[step] << " " << ens[step] << "\n";

            }
        cout << endl;
    }
    return 0;
}