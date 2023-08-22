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

For the range goals, the target need to be specified. This can either be a value, a parameter or a timeseries. 

The required columns of the `goal_table` are:

- `id`: A unique string for each goal.
- `active`: Either `0` or `1`. If `0` goal will not be used.
- `state`: State (variable) on which the goal should act on.
- `goal_type`: Choose from path goals: `range`,  `minimization_path` or `maximization_path`.
- `priority`: Priority of the goal.

And optional columns are:
- `function_min`: For goals of type `range` specify the minimum possible value for the selected state. 
- `function_max`: For goals of type `range` specify the maximum possible value for the selected state.
- `function_nominal`: Approximate order of the state.
- `target_data_type`: Either `value`, `parameter` or `timeseries`.
- `target_min`: Only for goals of type `range`: specify either a value or the name of the parameter/timeseries.
- `target_max`: Only for goals of type `range`: specify either a value or the name of the parameter/timeseries.
- `weight`: Weight of the goal.
- `order`: Only for goals of type `range`, order of the goal.'


To use to goal_generator, first import it as follows:

```python
from rtctools_interface.optimization.goal_generator_mixin import GoalGeneratorMixin
```

and add the `GoalGeneratorMixin` to your optimization problem class. It must be added before `GoalProgrammingMixin`. Also, define the `goal_table.csv` in the input folder of your problem.

### Notes
- The `minimization_path` and `maximization_sum` goals can be used to minimize/maximize the sum of a state over all timesteps, but be careful with the order:
    - For a `maximization_path` goal, if the order is even, the goal is equal to the minimization_path goal, as the minus sign is squared out.
    - A `minimization_path` or `maximization_path` goal with an even order will try to bring the selected state as close to 0 as possible, so not necessarily minimizing/maximizing it.

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
Then, add the `PlotGoalsMixin` to your optimization problem class. Set the class variable `plot_max_rows` to an integer number for the maximum number of rows. The number of columns will be derived from that. 
By default, the Mixin will look for the configuration table `input\plot_table.csv`. 

There are two types of plots that can be made with the PlotGoalsMixin
1. Plots based on goals in the goal_generator table
2. Plots of arbitrary states, for example ones being optimized in a goal defined in Python.

To add a plot for a goal in the `goal_generator` table, one should add a row to the `plot_table` with an `id` equal to the id of the goal in the `goal_generator` to be plotted. The `specified_in` field should be set to `goal_generator`.

To add a plot for a custom state, it is not necessary to set the `id`. One should set the `custom_state` to a state available in the model and set `specified_in` to `python`. For more the details, see the description of the columns below.

The (only) required column of this `plot_table` is:
- `y_axis_title`: A string (LaTeX allowed, between two `$`) for the y-axis.

And optional columns are:
- `id`: Required when a plot for a row in the `goal_table` should be created. Should be equal to the id in the corresponding `goal_table`.
- `variables_plot_1`: One or more state-names to be plotted, seperated by a comma. 
- `variables_plot_2`: One or more state-names to be plotted, seperated by a comma. Fixed styling is applied for all variables defined here.
- `custom_title`: Custom title overwriting automatic name. Required for goals specified in python.
- `specified_in`: Either `goal_generator` or `python`. If equal to `goal_generator`, the id field should be set. If equal to `python`, `custom_state` should be set.
- `custom_state`: A custom state to plot, only required and used in case `specified_in == 'python'`.


The table could thus look like:


|    id   |  y_axis_title   | variables_plot_1 | variables_plot_2 | custom_state | custom_title | specified_in
|---------|-----------------|------------------|------------------|------------------|------------------|------------------|
| goal_1  | Volume (\$m^3\$)  |      "PowerPlant1.QOut.Q"            |                  | | | goal_generator
| goal_2  | Volume (\$m^3\$)  |      "PowerPlant1.QOut.Q, PowerPlant2.QOut.Q"            |   | |               | goal_generator
|  | Volume (\$m^3\$)  |                |                  | electricity_cost | "Goal for minimizing electricity cost, at priority 10" | python


After running the model, in your output folder the folder `goal_figures` containing the figures is created.
