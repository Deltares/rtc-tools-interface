# Running a closed loop experiment
To run a closed loop experiment one can use the `run_optimization_problem_closed_loop` function from `run_closed_loop`.
This function is a drop-in replacement for the `run_optimization_problem` of rtc-tools. The user only needs to specify the `closed_loop_dates.csv` in the input folder of the optimization problem.

## Setup
Import `run_optimization_problem_closed_loop` with:
```python
from rtctools_interface.closed_loop.runner import run_optimization_problem_closed_loop
```
Add a table named `closed_loop_dates.csv` to the input folder of your optimization problem. The table should contain two columns: `start_date` and `end_date`.
Each row of the table corresponds to one modelling period. 

Example table `closed_loop_dates.csv`:
```
start_date, end_date
2024-05-19, 2024-05-23
2024-05-23, 2024-05-25
```
With this table rtc-tools will run two optimization problems (modelling periods): one with the data from 2024-05-19 upto and including 2024-05-23 and one from 2024-05-23 upto and including 2024-05-25. 
The `run_optimization_problem_closed_loop` will automatically set the final results from the previous as initial conditions of the next run. Note that this happens for:
- All variables available at the first time step in original timeseries_import, but not available at any timestep in the modelling period.
- All variables in the `initial_state.csv` (if the csv_mixin is used).

## Notes
- The first start_date in your `closed_loop_dates.csv` should be equal to the start_date of your timeseries_import.
- The different horizons should overlap with at least one day (to allow retrieving and setting initial values). An overlap of more days is allowed.
- Currently only a single timestep is copied as an initial value.
- The closed_loop runner only works in combination with the CSVMixin or the PIMixin. The CDFMixin is not supported.
