# rtc-tools-interface

This is rtc-tools-interface, a toolbox for user-interfaces for [rtc-tools](https://gitlab.com/deltares/rtc-tools).

## Install

```bash
pip install rtc-tools-interface
```

## Goal generator
The `goal generator` can be used to automatically add goals based on a csv file. Currently, the following goal types are supported:
- range (default order is 2)
- minimization_path (default order is 1)
- maximization_path (default order is 1)
- minimization_ramping (default order is 2)
- range_rate_of_change (default order is 1)

For the range goals, the target need to be specified. This can either be a value, a parameter or a timeseries.

The required columns of the `goal_table` are:

- `id`: A unique string for each goal.
- `active`: Either `0` or `1`. If `0` goal will not be used.
- `state`: State (variable) on which the goal should act on.
- `goal_type`: Choose from path goals: `range`,  `minimization_path`, `maximization_path`, `minimization_ramping` or `range_rate_of_change`.
- `priority`: Priority of the goal.

And optional columns are:
- `function_min`: For goals of type `range` specify the minimum possible value for the selected state.
- `function_max`: For goals of type `range` specify the maximum possible value for the selected state.
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
$$  
\begin{aligned}
g_{low}(\epsilon_t) \leq &x_t \leq g_{up}(\epsilon_t) \quad &\forall t\\
 0 \leq &\epsilon_t \leq 1 \quad &\forall t 
\end{aligned}
$$
where
$$
\begin{aligned}
g_{low}(\epsilon_t) &:= (1-\epsilon_t) m_{t,target} + \epsilon_t m \\
g_{up}(\epsilon_t)  &:= (1-\epsilon_t) M_{t,target} + \epsilon_t M 
\end{aligned}
$$
and $w$ is equal to the `weight` (default is 1), $r$ is equal to the `order` (default is 2), $t$ the timestep, $x$ the selected `state`, $m_{target}$ and $M_{target}$ the lower and upper targets (`target_min` and `target_max`), $m$ and $M$ the actual bounds of $x$ (`function_min` and `function_max`). The auxiliary variable $\epsilon$ is automatically created by rtc-tools. In loose terms, the range goal tries to archieve
$$ m_{t,target} \leq x_t \leq M_{t,target} \quad \forall t$$
by minimizing, if any, the sum of exceedances for the timesteps. For more details on the range goal, see [Read the Docs](https://rtc-tools.readthedocs.io/en/latest/optimization/goal_programming/goals.html) of rtc-tools.

#### minimization_ramping
For the minimization_ramping goal, rtc-tools adds the following equation to the objective function of the specified priority
$$w\sum_{t} (x_t-x_{t-1})^r $$
where $w$ is equal to the `weight` (default is 1), $r$ is equal to the `order` (default is 2), $t$ the timestep and $x$ the selected `state`. Note that for the derivative of the first timestep the initial value of $x$ is used. No constraints are added for this goal. 

#### range_rate_of_change
The range_rate_of_change is constructed as a range goal. The only difference is that $x_t$ is replaced by $der(x_t)$.

### Example goal table
See the table below for an example content of the `goal_table.csv`.

| id     | state | active | goal_type    | function_min | function_max | function_nominal | target_data_type | target_min | target_max | priority | weight | order |
|--------|-------|--------|--------------|--------------|--------------|------------------|------------------|------------|------------|----------|--------|-------|
| goal_1 | reservoir_1_waterlevel     | 1      | range        | 0            | 15           | 10               | value            | 5.0        | 10.0       | 5       |        |       |
| goal_2 | reservoir_2_waterlevel     | 1      | range        | 0            | 15           | 10               | timeseries            | "target_series"        | "target_series"       | 10       |        |       |
| goal_3 | electricity_cost     | 1      | minimization_path |              |              |                  |                  |            |            | 20       |        |       |

## Plot resuls after each priority
By using the `PlotGoalsMixin`, plots will be generated after optimizing for each unique priority. To utilize this functionality, import the mixin as follows:
```python
from rtctools_interface.optimization.plot_goals_mixin import PlotGoalsMixin
```
Then, add the `PlotGoalsMixin` to your optimization problem class. Note that, for now, `PlotGoalsMixin` only works in combination with the `GoalGeneratorMixin`. Include the `PlotGoalsMixin` before the `GoalGeneratorMixin`!

Set the class variable `plot_max_rows` to an integer number for the maximum number of rows (default is 4). The number of columns will be derived from that.
By default, the Mixin will look for the configuration table `input\plot_table.csv`.

There are two types of plots that can be made with the PlotGoalsMixin
1. Plots based on goals in the goal_generator table
2. Plots of arbitrary states, for example ones being optimized in a goal defined in Python.

To add a plot for a goal in the `goal_generator` table, one should add a row to the `plot_table` with an `id` equal to the id of the goal in the `goal_generator` to be plotted. The `specified_in` field should be set to `goal_generator`.

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


After running the model, in your output folder the folder `goal_figures` containing the figures is created.
