model Example
  // Strongly simplified version of the Electra boezem (rtc-electra):
  // one linear storage basin that receives polder discharges and is
  // drained to the Waddenzee by two pumping stations (KGM155 and Den Deel).

  // Storage basin
  parameter Real A = 2.0e6;          // Wetted surface area [m2]
  parameter Real H_b = -3.0;         // Bottom level [m NAP]

  // Relative energy use per m3 of the pumping stations (Den Deel is less efficient)
  parameter Real eff_KGM155 = 1.0;
  parameter Real eff_DenDeel = 1.3;

  // Boundary conditions (read from timeseries_import.csv)
  input Real Q_polders(fixed = true);
  input Real price(fixed = true);

  // Control variables
  input Real Q_KGM155(fixed = false, min = 0.0, max = 33.5, nominal = 10.0);
  input Real Q_DenDeel(fixed = false, min = 0.0, max = 10.6, nominal = 5.0);

  // States
  Real V(min = 0.0, nominal = 4.0e6);
  output Real H(min = -1.2, max = -0.6);
  output Real pump_cost;
equation
  der(V) = Q_polders - Q_KGM155 - Q_DenDeel;
  H = H_b + V / A;
  pump_cost = price * (eff_KGM155 * Q_KGM155 + eff_DenDeel * Q_DenDeel);
end Example;

