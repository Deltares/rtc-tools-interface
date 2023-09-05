model BasicModel
	input Real u(fixed=false, min=-10, max=15);
	input Real f(fixed=true, min=-10, max=15);
	parameter Real scale = 3600;
	output Real x();

equation
	der(x) = (-u + f) / scale;

end BasicModel;
