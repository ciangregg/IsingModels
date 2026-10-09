#include <iostream>

#include <cmath>
#include <random>
#include <vector>

#include <fstream> // output
#include <sstream>// for string
#include <iomanip>// for string


using namespace std;

#define L 100
#define N (L*L)
#define XNN 1
#define YNN L

const int J = 1;

int s[N];
double prob[5];
double T;

// works better than my attempt, this uses Mersenne Twister
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

        if (delta <= 0)
            s[i] = -s[i];
        else if (drandom() < prob[delta])
            s[i] = -s[i];
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
    T = 2.0;     
    
    const int nsweeps = 100000;
    int  printingNo=nsweeps/100;
    vector<int> mags(nsweeps), ens(nsweeps);
    
    // start below T_c approx 2.69 
    for (int i = 0; i < N; i++) {   // start with all spins up
        s[i] = 1;
    }

    initialise();

    cout << "T=" << T << "  prob[2]=" << prob[2] << "  prob[4]=" << prob[4] << endl;

    for(int step = 0; step < nsweeps; step++) {
        sweep();
        ens[step] = energy();
        mags[step] = magnetisation();
           if (step % printingNo == 0){
            cout << "\r" << 100.0 * step / nsweeps << "% of " << nsweeps << flush;
            }
    }
    cout<<endl;

    string filename ="data/met_results_T" + rounding(T,1) +"L"+to_string(L) + "nsweeps"+to_string(nsweeps)+".dat";
    ofstream file(filename);
    for (int step = 0; step < nsweeps; step++)
        file << step << " " << mags[step] << " " << ens[step] << "\n";

    return 0;
}