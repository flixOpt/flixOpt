# flixOpt
vector based energy and material flow optimization framework in python

flixOpt is in early state! Not completely documented yet,... Do not hesitate to contact authors! 

Collaboration is welcome!
## introduction
**flixOpt** is an vector based optimization framework creating and solving mixed-integer programming problems (MILP). It is created with focus on energy flows but can be used for material flows as well.

flixOpt was developed in project SMARTBIOGRID by [TU Dresden](https://github.com/gewv-tu-dresden).
This project was funded by the German Federal Ministry for Economic Affairs and Energy (FKZ: 03KB159B).

flixOpt development is based on matlab framework flixOptMat developed in project FAKS by [TU Dresden](https://github.com/gewv-tu-dresden) and has a few influences from [oemof/solph](https://github.com/oemof/oemof-solph) (Great thanks for your tool!)

## Usage
Install this package via pip in to your environment: `pip install git+https://github.com/flixOpt/flixOpt.git`

## key features
  * various constraints available
  * operation optimization optionally combined with investment optimization
  * segmented linear correlations for
    * flows
    * invest costs and invest size
  * effects 
    * various effects, i.g. costs, CO2 emissions, primary energy, area demand etc.
    * effects coupleable, i.g. specific costs of CO2-emissions
    * constraints, i.g. max sum of CO2 emissions
    * simply switch effect, which should be minimized (optimization target)
  * others
    * non-equidistant timesteps possible
    * investment and flow-on/off variables in one model
## performance issues
You can choose between three calculation modes:
  * **full** -> exact and slow
  * **segmented** (with variable time overlap) -> fast but not exact for big storages
  * **aggregated** (automatically creation of typical periods via [TSAM](https://github.com/FZJ-IEK3-VSA/tsam "more info")) -> fast, quite exact
## architecture
  * interlayer flixBase for modeling and good overview of (vectorized) variables and equations
  * postprocessing unit  
  * allows integration of other modeling languages than [Pyomo](http://www.pyomo.org/)
<img src="/pics/architecture_flixOpt.png" style=" height:400px "  >

## solver
  * You need to install a solver. Various solvers are usable. Recommended opensource solvers are [CBC](https://github.com/coin-or/Cbc) and [GLPK](https://www.gnu.org/software/glpk/). Executables can be found for example  [here for CBC](https://portal.ampl.com/dl/open/cbc/) and [here for GLPK](https://sourceforge.net/projects/winglpk/) (Windows: You have to put solver-executables to the PATH-variable)

## citing
For explicitly citing, a link to a paper is coming soon ...

Temporarily use <https://doi.org/10.13140/RG.2.2.14948.24969>

## developement
 * Testing: Using `python -m unittest discover -s tests` in the commant line will run all tests
