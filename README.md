# rtc-tools-interface

This is rtc-tools-interface, a toolbox for user-interfaces for [rtc-tools](https://gitlab.com/deltares/rtc-tools).

## Install

```bash
pip install rtc-tools-interface
```

## Goal generator
The `goal generator` can be used to automatically add goals based on a csv file. Currently, the following goal types are supported:
- range
- minimization

For the range goals, the target need to be specified. This can either be a value, a parameter or a timeseries. 

The `goal_table` should have the following columns:

- `id`: A unique string for each goal.
- `state`: State (variable) on which the goal should act on.
- `goal_type`: Choose from path goals: `range`,  `minimization` or `maximization`, or goals: `minimization_sum` or `maximization_sum`.
- `function_min`: For goals of type `range`, `minimization_sum` or `maximization_sum, specify the minimum possible value for the selected state. 
- `function_max`: For goals of type `range`, `minimization_sum` or `maximization_sum, specify the maximum possible value for the selected state.
- `function_nominal`: Approximate order of the state.
- `target_data_type`: Either `value`, `parameter` or `timeseries`.
- `target_min`: Only for goals of type `range`, `minimization_sum` or `maximization_sum`: specify either a value or the name of the parameter/timeseries.
- `target_max`: Only for goals of type `range`, `minimization_sum` or `maximization_sum`: specify either a value or the name of the parameter/timeseries.
- `priority`: Priority of the goal.
- `weight`: Weight of the goal.
- `order`: Only for goals of type `range`, order of the goal.

The `minimization_sum` and `maximization_sum` goals are targte goals on the sum of a state over the optimization time horizon. Therefore, to minimize the sum, only a target_max is required (as small as possible), and for a maximization goal only a target_min is required (as large as possible). If the user does not provide these, these wil be calculated from the function bounds. If the user does not provide function bounds, these will be calculated from the bounds of the state. Note that the user needs to at least provide the function range if the state is unbounded.

To use to goal_generator, first import it as follows:

```python
from rtctools_interface.optimization.goal_generator_mixin import GoalGeneratorMixin
```

and add the `GoalGeneratorMixin` to your optimization problem class. It must be added before `GoalProgrammingMixin`. Also, define the `goal_table.csv` in the input folder of your problem.

### Example goal table
See the table below for an example content of the `goal_table.csv`. 

| id     | state | active | goal_type    | function_min | function_max | function_nominal | target_data_type | target_min | target_max | priority | weight | order |
|--------|-------|--------|--------------|--------------|--------------|------------------|------------------|------------|------------|----------|--------|-------|
| goal_1 | reservoir_1_waterlevel     | 1      | range        | 0            | 15           | 10               | value            | 5.0        | 10.0       | 5       |        |       |
| goal_2 | reservoir_2_waterlevel     | 1      | range        | 0            | 15           | 10               | timeseries            | "target_series"        | "target_series"       | 10       |        |       |
| goal_3 | electricity_cost     | 1      | minimization |              |              |                  |                  |            |            | 20       |        |       |

## Plot resuls after each priority
By using the `PlotGoalsMixin`, plots will be generated after optimizing for each unique priority. To utilize this functionality, import the mixin as follows:
```python
from rtctools_interface.optimization.plot_goals_mixin import PlotGoalsMixin
```
Then, add the `PlotGoalsMixin` to your optimization problem class. Set the class variable `plot_max_rows` to an integer number for the maximum number of rows. The number of columns will be derived from that. By default, the Mixin will look for the configuration table `input\plot_table.csv`. This table should contain the columns `id`, `y_axis_title`, `variables_plot_1` and `variables_plot_2`. Each `id` in the `id` column should also appear in the table defining the goals. The table could thus look like:

|    id   |  y_axis_title   | variables_plot_1 | variables_plot_2 |
|---------|-----------------|------------------|------------------|
| goal_1  | Volume (\$m^3\$)  |      "PowerPlant1.QOut.Q"            |                  |
| goal_2  | Volume (\$m^3\$)  |      "PowerPlant1.QOut.Q, PowerPlant2.QOut.Q"            |                  |


After running the model, in your output folder the folder `goal_figures` containing the figures is created.
