# rtc-electra plot example (rtc-tools-interface)

Small, self-contained example based on `rtc-electra` that shows how to use
[rtc-tools-interface](https://github.com/Deltares/rtc-tools-interface) to plot
**optimization** and **simulation** results.

The Electra boezem is reduced to one linear storage basin (`H = H_b + V / A`) that
receives polder discharge (`Q_polders`) and is drained by two pumping stations
(`Q_KGM155`, `Q_DenDeel`). The energy price (`price`) is lower at night.

## Structure

```
model/Example.mo             optimization model
model/ExampleSim.mo          simulation model (pumps set from Python)
input/timeseries_import.csv  Q_polders, price, H_min, H_max (48 h, hourly)
input/initial_state.csv      initial storage volume V
input/goal_table.csv         goals read by GoalGeneratorMixin
input/plot_table.csv         plot configuration for the optimization
input/plot_table_sim.csv     plot configuration for the simulation
src/example_opt.py           optimization (PlotMixin + GoalGeneratorMixin)
src/example_sim.py           rule-based simulation (simulation PlotMixin)
```

## Run

```powershell
cd src
python example_opt.py   # runs the optimization; its post() also runs the simulation
python example_sim.py   # (optional) run only the simulation
```

## Output

| Run          | Figures                                                                |
|--------------|------------------------------------------------------------------------|
| optimization | `output/optimization/figures/after_priority_10.html`, `after_priority_20.html`, `final_results.html` |
| simulation   | `output/simulation/figures/final_results.html`                         |

Also written: `timeseries_export.csv`, `cached_results/*.json` (previous run, used for the
"show/hide results from previous run" button) and, for the optimization,
`performance_metrics/<goal id>.csv`.

Use `plotting_library="matplotlib"` in the scripts to get `.png` figures instead of `.html`.

## How the plotting is configured

**Goal table** (`goal_table.csv`): goal `1` keeps `H` between the `H_min`/`H_max`
timeseries (priority 10), goal `2` minimizes `pump_cost` (priority 20).

**Plot table** (`plot_table.csv`, `plot_table_sim.csv`), one row per subplot:

| column                           | meaning                                                         |
|----------------------------------|-----------------------------------------------------------------|
| `id`                             | goal id (only for `specified_in = goal_generator`)              |
| `y_axis_title`, `custom_title`   | axis label and subplot title                                    |
| `variables_style_1`              | variables plotted as normal lines                              |
| `variables_style_2`              | variables plotted as thin lines                                |
| `variables_with_previous_result` | also shows the result of the previous priority (optimization)  |
| `specified_in`                   | `goal_generator`: plot the goal state + target bounds; `python`: plot listed variables only |

The simulation plot table can only use `python` rows, and its variables must be model
variables. That is why `H_min`/`H_max` are inputs of `ExampleSim.mo`.

