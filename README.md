# flixOpt
vector based energy and material flow optimization framework in python
## introduction
**flixOpt** is an vector based optimization framework creating and solving mixed-integer programming problems (MILP). It is created with focus of energy flows but can be used for material flows as well.
in progress; do not hesitate to contact authors

flixOpt is developed in project SMARTBIOGRID by [TU Dresden](https://github.com/gewv-tu-dresden)

flixOpt is based on matlab framework flixOptMat developed in project FAKS by [TU Dresden](https://github.com/gewv-tu-dresden) and has a few influences from [oemof/solph](https://github.com/oemof/oemof-solph) (Great thanks for your tool!)
## key features
  * architecture allows integration of other modeling languages than [Pyomo](http://www.pyomo.org/)
  * many constraints available
  * invest optimization
  * segmented linear correlations for
    * flows
    * invest costs and invest size
  * effects 
    * various effects, i.g. costs, CO2 emissions, primary energy, area demand etc.
    * effects coupleable, i.g. specific costs of CO2-emissions
    * constraints, i.g. max sum of CO2 emissions

## performance issues
You can choose between three calculation modes:
  * **full** -> exact and slow
  * **segmented** (with variable time overlap) -> fast but not exact for big storages
  * **aggregated** (automatically creation of typical periods via [TSAM](https://github.com/FZJ-IEK3-VSA/tsam "more info")) -> fast, quite exact
## citing
For explicitly citing, a link to a paper is coming soon ...

Temporarily use <https://doi.org/10.13140/RG.2.2.31085.87527>
