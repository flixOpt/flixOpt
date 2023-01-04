# flixOpt
vector based energy and material flow optimization framework in python

flixOpt is in early state! Not completely documented yet,... Do not hesitate to contact authors! 

Collaboration is welcome!
## introduction
**flixOpt** is an vector based optimization framework creating and solving mixed-integer programming problems (MILP). It is created with focus on energy flows but can be used for material flows as well.

flixOpt was developed in project SMARTBIOGRID by [TU Dresden](https://github.com/gewv-tu-dresden)

flixOpt is based on matlab framework flixOptMat developed in project FAKS by [TU Dresden](https://github.com/gewv-tu-dresden) and has a few influences from [oemof/solph](https://github.com/oemof/oemof-solph) (Great thanks for your tool!)
## key features
  * various constraints available
  * invest optimization
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
  * interlayer flixBase for modeling and good overview of variables and equations
  * postprocessing unit  
  * allows integration of other modeling languages than [Pyomo](http://www.pyomo.org/)
<img src="/pics/architecture_flixOpt.png" style=" height:400px "  >

## citing
For explicitly citing, a link to a paper is coming soon ...

Temporarily use <https://doi.org/10.13140/RG.2.2.14948.24969>
