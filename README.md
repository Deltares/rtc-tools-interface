# rtc-tools-interface

This is rtc-tools-interface, a toolbox for user-interfaces for [rtc-tools](https://gitlab.com/deltares/rtc-tools).

## Install

```bash
pip install rtc-tools-interface
```

## Table of Contents
1. [Goal generator](#goal-generator)
2. [Goal performance metrics](#goal-performance-metrics)
3. [Automatic plotting of results](#automatic-plotting-of-results)
4. [Closed loop runner](#closed-loop-runner)

## Goal generator
The `goal generator` can be used to automatically add goals based on a csv file. Currently, the following goal types are supported:
- range (default order is 2)
- minimization_path (default order is 1)
- maximization_path (default order is 1)
- range_rate_of_change (default order is 1)

For the range goals, the target need to be specified. This can either be a value, a parameter or a timeseries.

The required columns of the `goal_table` are:

- `id`: A unique string for each goal.
- `active`: Either `0` or `1`. If `0` goal will not be used.
- `state`: State (variable) on which the goal should act on.
- `goal_type`: Choose from path goals: `range`,  `minimization_path`, `maximization_path` or `range_rate_of_change`.
- `priority`: Priority of the goal.

And optional columns are:
- `function_min`: For goals of type `range` specify the minimum possible value for the selected state. If not specified, will be calculated using the bounds of the `state` (if available).
- `function_max`: For goals of type `range` specify the maximum possible value for the selected state. If not specified, will be calculated using the bounds of the `state` (if available).
- `function_nominal`: Approximate order of the state.
- `target_data_type`: Either `value`, `parameter` or `timeseries`.
- `target_min`: Only for goals of type `range`: specify either a value or the name of the parameter/timeseries.
- `target_max`: Only for goals of type `range`: specify either a value or the name of the parameter/timeseries.
- `weight`: Weight of the goal.
- `order`: Only for goals of type `range`, order of the goal.


To use to goal_generator, first import it as follows:

```python
from rtctools_interface.optimization.goal_generator_mixin import GoalGeneratorMixin
```

and add the `GoalGeneratorMixin` to your optimization problem class. It must be added before `GoalProgrammingMixin`. Also, define the `goal_table.csv` in the input folder of your problem.

### Notes
- The `minimization_path` and `maximization_path` goals can be used to minimize/maximize the sum of a `state` over all timesteps, but be careful with the order:
    - For a `maximization_path` goal, if the order is even, the goal is equal to the minimization_path goal, as the minus sign is squared out.
    - A `minimization_path` or `maximization_path` goal with an even order will try to bring the selected `state` as close to 0 as possible, so not necessarily minimizing/maximizing it.

### Goal equations
For each goal, this section will specify the equations that rtc-tools will add to the optimization problem. Note that rtc-tools will always _minimize_ the objective function.
#### minimization_path
For the minimization_path goal, rtc-tools adds the following equation to the objective function of the specified priority
$$w\sum_t x_t^r $$
where $w$ is equal to the `weight` (default is 1), $r$ is equal to the `order` (default is 1), $t$ the timestep and $x$ the selected `state`. No constraints are added for this goal. 

#### maximization_path
For the maximization_path goal, rtc-tools adds the following equation to the objective function of the specified priority
$$w\sum_t (-x_t)^r $$
where $w$ is equal to the `weight` (default is 1), $r$ is equal to the `order` (default is 1), $t$ the timestep and $x$ the selected `state`. No constraints are added for this goal. 
#### range
For the range goal, rtc-tools adds the following equation to the objective function of the specified priority
$$w\sum_t \epsilon_t^r $$
and the following constraints

$$g_{low}(\epsilon_t) \leq x_t \leq g_{up}(\epsilon_t) \quad \forall t $$

$$0 \leq \epsilon_t \leq 1 \quad \forall t $$

where

$$g_{low}(\epsilon_t) = (1-\epsilon_t) m_{t,target} + \epsilon_t m $$

$$g_{up}(\epsilon_t)  = (1-\epsilon_t) M_{t,target} + \epsilon_t M $$

and $w$ is equal to the `weight` (default is 1), $r$ is equal to the `order` (default is 2), $t$ the timestep, $x$ the selected `state`, $m_{target}$ and $M_{target}$ the lower and upper targets (`target_min` and `target_max`), $m$ and $M$ the actual bounds of $x$ (`function_min` and `function_max`). The auxiliary variable $\epsilon$ is automatically created by rtc-tools. 

In loose terms, the range goal tries to archieve

$$m_{t,target} \leq x_t \leq M_{t,target} \quad \forall t$$

by minimizing, if any, the sum of exceedances for the timesteps. For more details on the range goal, see [Read the Docs](https://rtc-tools.readthedocs.io/en/latest/optimization/goal_programming/goals.html) of rtc-tools.


#### range_rate_of_change
The range_rate_of_change goal can be used to set a target range on ramp rate for a model state. Like the range goal, one needs to set the `target_min` and `target_max` for that. **Importantly** for the range_rate_of_change goal, the supplied values are relative to the nominal of the function. So supplying a `target_max` of `10` corresponds to the aim of having a maximum increase per timestep of 10% * `nominal`, where the nominal automatically set to `maximum rate of change`/2 or specified manually. To formulate the target of having a maximum increase and decrease by of 10% per timestep, one would set the `target_min` to `-10` and the `target_max` to `10`.

The equations for the range_rate_of_change goal are almost the same as for the range goal, which can be found above. The only difference is that $x_t$ is replaced by the derivative of $x$ with respect to $t$, calculated as $\frac{x_t - x_{t-1}}{\Delta t}$. RTC-Tools computes the derivative (the ramp rate) internally for the state variable specified in the goal table. 

### Example goal table
See the table below for an example content of the `goal_table.csv`.

| id     | state | active | goal_type    | function_min | function_max | function_nominal | target_data_type | target_min | target_max | priority | weight | order |
|--------|-------|--------|--------------|--------------|--------------|------------------|------------------|------------|------------|----------|--------|-------|
| goal_1 | reservoir_1_waterlevel     | 1      | range        | 0            | 15           | 10               | value            | 5.0        | 10.0       | 5       |    1    |    1   |
| goal_2 | reservoir_2_waterlevel     | 1      | range        | 0            | 15           | 10               | timeseries            | "target_series"        | "target_series"       | 10       |    1    |   1    |
| goal_3 | electricity_cost     | 1      | minimization_path |              |              |                  |                  |            |            | 20       |   1     |   1    |
| goal_4 | reservoir_2_volume | 1 | range_rate_of_change | -100 | 100 | 10 | value | -10 | 10 | 30 | 1 | 1 |  

## Goal performance metrics
For all goals defined with the goal generator this rtc-tools-interface module will also calculate performance metrics. By default, these performance metrics are saved to a .csv in the folder `output/perfomance_metrics`, with one csv file per goal. With the class variable `calculate_performance_metrics` this functionality can be disabled (by default it is enabled).

The calculated metrics are:
- `timeseries_sum`: The sum of the state variable over all timesteps.
- `timeseries_min`: The minimum of the state variable.
- `timeseries_max`: The maximum of the state variable.
- `timeseries_avg`: The average of the state variable.
- `max_difference`: The maximum difference in one timestep.
- `mean_absolute_percentual_difference`: The mean of the absolute percentual difference per timestep over all timesteps (only for range goals).
- `mean_absolute_difference`: The mean absolute difference per timestep of the state variable over all timesteps (only for range goals).


## Automatic plotting of results
With the `PlotMixin` one can easily make plots of the results of rtc-tools. This functionality can be used both for optimization and simulation problems. For optimization problems, use:
```python
from rtctools_interface.optimization.plot_mixin import PlotMixin
```
and for simulation problems use:
```python
from rtctools_interface.simulation.plot_mixin import PlotMixin
```
Then, add the `PlotMixin` to your optimization/simulation problem class. For optimization problems, the PlotMixin can create a plot after each priority and/or a plot with the final results only.
By default, the `PlotMixin` will make both. This can be changed by setting the class variables `plot_results_each_priority` and `plot_final_results` to either `True` or `False` in your problem class.

Furthermore, the PlotMixin can either create `Plotly` plots and `matplotlib` plots. The `matplotlib` plots will be exported as `png`, the Plotly figures as `html`. By default, `Plotly` is used. To change this, pass the keyword-argument `plotting_library="matplotlib"` to the `run_optimization_problem` function.

### Comparing results from different runs
- In optimization mode, the plots for a particular priority will contain line segments with the results from the previous priority result. This makes it easy to see what changed from priority to priority.
- The `final_results` plot will show the result from the **previous run**. This allows for comparing results from different scenario's (like input timeseries or changes to the model). Note that is not possible to change the number of goals between two comparison runs. This feature currently only works with Plotly plots, where a dropdown is available to hide the previous results.
### Configuration variables
The following class variables can be set to change the behaviour of the PlotMixin:
- `plot_max_rows`: an integer number for the maximum number of rows (default is 4). The number of columns will be derived from that.
- `plot_results_each_priority`: Only for optimization: boolean indicating whether the plots for each priority should be generated and saved. Default is True.
- `plot_table_file`: path to plot table csv file. Default is `input\plot_table.csv`.

### Specifying the plot table
There are two types of plots that can be made with the PlotMixin
1. Plots of arbitrary states, for example ones being optimized in a goal defined in Python.
2. Plots based on goals in the goal_generator table (only applicable to optimization problems).

To add a plot for a goal in the `goal_generator` table, one should add a row to the `plot_table` with an `id` equal to the id of the goal in the `goal_generator` to be plotted. The `specified_in` field should be set to `goal_generator`. Rows of the `plot_table` with `specified_in`=`goal_generator` but with an `id` that does not occcur in the `goal_table`, are ignored.  

To add a plot for a custom state, it is not necessary to set the `id`. However, by default no variables will be plotted. To do so, one needs to specify at least one variable.

The (only) required column of this `plot_table` is:
- `y_axis_title`: A string for the y-axis (LaTeX allowed, between two `$`).

And optional columns are:
- `id`: Required when a plot for a row in the `goal_table` should be created. Should be equal to the id in the corresponding `goal_table`.
- `variables_style_1`: One or more state-names to be plotted, seperated by a comma.
- `variables_style_2`: One or more state-names to be plotted, seperated by a comma. Fixed styling is applied for all variables defined here.
- `variables_with_previous_result`: One or more state-names to be plotted, seperated by a comma. If available, the results for that variable at the previous priority optimization will also be shown.
- `custom_title`: Custom title overwriting automatic title. Required for goals specified in python.
- `specified_in`: Either `goal_generator` or `python`. If equal to `goal_generator`, the id field should be set.


The table could thus look like:


|    id   |  y_axis_title   | variables_style_1 | variables_style_2 | variables_with_previous_result | custom_title | specified_in
|---------|-----------------|------------------|------------------|------------------|------------------|------------------|
| goal_1  | Volume (\$m^3\$)  |      "PowerPlant1.QOut.Q"            |                  | | | goal_generator
| goal_2  | Volume (\$m^3\$)  |      "PowerPlant1.QOut.Q, PowerPlant2.QOut.Q"            |   | |               | goal_generator
|  | Volume (\$m^3\$)  |                |                  | electricity_cost | "Goal for minimizing electricity cost, at priority 10" | python


After running the model, in your output folder the folder `figures` containing the figures is created.

## Closed loop runner

To run a closed loop experiment
one can use the `run_optimization_problem_closed_loop` function
from `rtctools_interface.closed_loop.runner`.
This function is a drop-in replacement for the `run_optimization_problem` of rtc-tools.
The user needs to specify a `ClosedLoopConfig` configuration
from `rtctools_interface.closed_loop.config`
to specify the time ranges for which to subsequentially solve the optimization problem.

### Setup

Import `ClosedLoopConfig` and `run_optimization_problem_closed_loop` with:

```python
from rtctools_interface.closed_loop.config import ClosedLoopConfig
from rtctools_interface.closed_loop.runner import run_optimization_problem_closed_loop
```
#### Fixed inputs
Create the file `fixed_inputs.json`, in which you specify which variables in your
timeseries import are what we call 'fixed_inputs'. They are timeseries that the closed
loop runner should simply copy as they are, even if they contain only NaNs in a modelling
period.

The variables that are not mentioned in this list of fixed_inputs, and have only NaN's
in a modelling period, are considered being 'initial values'. The closed loop runner will set 
the first timestep of each modelling period with the corresponding calculated value from the 
previous modelling period.


#### Closed loop config
A `ClosedLoopConfig` configuration can be created from a csv file or
from a given forecast timestep (time between each time range)
and optimization period (duration of each time range).
An option `round_to_dates` can be used to round the start and end time of each range to a date,
i.e. the start time is rounded to the start of the day
and the end time is rounded to the end of a day.
Examples of creating a configuration are given below.

```python
from datetime import timedelta

from rtctools_interface.closed_loop.config import ClosedLoopConfig

config_from_file = ClosedLoopConfig(
    file="path/to/closed_loop_dates.csv",
    round_to_dates=True
)

config_from_fixed_periods = ClosedLoopConfig.from_fixed_periods(
    optimization_period=timedelta(days=3),
    forecast_timestep=timedelta(days=2)
)
```

The CSV file `closed_loop_dates.csv` has two columns `start_date` and `end_date`
and looks as follows.

```
start_date, end_date
2024-05-19, 2024-05-23
2024-05-23, 2024-05-25
```

The `run_optimization_problem_closed_loop` will solve the optimization problem
for each time range subseqentually.
It will use the final results from the previous run to set the initial values of the next run.
Note that this happens for:

- All variables available at the first time step in original timeseries_import,
    but not available at any timestep in the modelling period.
- All variables in the `initial_state.csv` (if the csv_mixin is used).

### Notes

- The start time of the first time range should coincide
    with the start time of the input timeseries.
- The start time of the next time range should be less or equal
    to the end time of the current time range.
- Currently, only the initial values of the first time step in a given time range are set.
- The closed_loop runner only works in combination with the CSVMixin or the PIMixin.
    The CDFMixin is not supported.

### Matplotlib in Headless Environments

For CI or servers without a display, set `MPLBACKEND=Agg` to ensure Matplotlib uses a non-interactive backend:

```bash
export MPLBACKEND=Agg
```
