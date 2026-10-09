#include <iostream>

#include <cmath>
#include <random>
#include <vector>


#include <fstream> // output
#include <sstream>// for string
#include <iomanip>// for string
#include <filesystem> // similar to os in python



using namespace std;
namespace fs = std::filesystem;


#define L 101
#define N (L*L)
#define XNN 1
#define YNN L

const int J = 1;

int s[N];
double prob[5];
double T;
const double Tc= 2 * abs(J) / log(1 + sqrt(2));
int curM, curE;      // running totals

// <random>, works better than my attempt, this uses Mersenne Twister
mt19937 gen(12345);
uniform_real_distribution<double> dist(0.0, 1.0);

double drandom() { return dist(gen); }

/*
double drandom(){
    double randomNum100 = rand() % 101;
    return randomNum100/100;
}
*/
void initialise() {
    for (int i = 2; i < 5; i += 2) {
        prob[i] = exp(-(2.0 * i )/ T);
    }
}

void sweep() {
    for (int k = 0; k < N; k++) {
        int i = N * drandom();
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
        if(right>= N){
            right-=N;
        }

        int up = i + YNN;
        if(up>= N){
            up-=N;
        }

        E -= J*s[i]*(s[right]+s[up]);
    };
    return E;
}

string rounding(double x, int digits = 3) {
    ostringstream s;
    s << fixed << setprecision(digits) << x;
    return s.str();
}

int main() {
    double nT=50; // number of temps
    double Tmin=1.8;
    double Tmax=2.8;

    const int nsweeps = 10000;
    int  printingNo=nsweeps/100;

    vector<int> mags(nsweeps), ens(nsweeps);
    
     for (int t = 0; t < nT; t++) {

        T = Tmin + (Tmax - Tmin) * t / (nT - 1);
        
        initialise();

        cout << "T=" << T << "  prob[2]=" << prob[2] << "  prob[4]=" << prob[4] << endl;

        // start below T_c approx 2.69
        if(T< Tc){
            for (int i = 0; i < N; i++) {   // start with all spins up
                s[i] = 1;
            }
        }
        else{
            //random assignment of spins s[i]
            for (int i = 0; i < N; i++) {   
                if(drandom()<0.5){
                s[i] = 1;
                }
                else{
                s[i]= -1;
                }
            }

        }
        for(int step = 0; step < nsweeps; step++) {
            curM = magnetisation();  
            curE = energy(); 

            sweep();
            ens[step] = energy();
            mags[step] = magnetisation();
            if (step % printingNo == 0){
                cout << "\r" << 100.0 * step / nsweeps << "% of " << nsweeps << flush;
                }
        }
        cout<<endl;

        string filename = "data/met_results/L" + to_string(L) + "/nsweeps" +
                  to_string(nsweeps) + "/T" +
                  rounding(T, 5) + ".dat";
        fs::create_directories(fs::path(filename).parent_path());
        ofstream file(filename);

        for (int step = 0; step < nsweeps; step++){
            file << step << " " << mags[step] << " " << ens[step] << "\n";
        }
        //string per=rounding(100*T/nT,3);
        cout << "finished T = " << T << endl;
    
    }
    return 0;
}