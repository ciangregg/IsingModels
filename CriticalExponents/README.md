
## Structure
```
CriticalExponents/

    README.md  
    data/              
    figs/  

    wolff.cpp          copy of efficient/wolff.cpp, main() takes arguments
    fss_tools.py       reused helpers: make_temps, loader, peak, fits
    fss_run.py         compile per L, write temperature files, run in parallel
    fss_analyse.py     pyerrors, extract exponents, plots  
```          