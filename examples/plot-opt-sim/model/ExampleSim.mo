model ExampleSim
  // Same system as Example.mo, used for a rule-based simulation.
  // The pump discharges are set by a simple on/off controller in Python.

  parameter Real A = 2.0e6;
  parameter Real H_b = -3.0;
  parameter Real eff_KGM155 = 1.0;
  parameter Real eff_DenDeel = 1.3;

  // Boundary conditions (read from timeseries_import.csv)
  input Real Q_polders;
  input Real price;
  input Real H_min;
  input Real H_max;

  // Set by the controller in example_sim.py
  input Real Q_KGM155;
  input Real Q_DenDeel;

  Real V;
  output Real H;
  output Real pump_cost;
equation
  der(V) = Q_polders - Q_KGM155 - Q_DenDeel;
  H = H_b + V / A;
  pump_cost = price * (eff_KGM155 * Q_KGM155 + eff_DenDeel * Q_DenDeel);
end ExampleSim;

