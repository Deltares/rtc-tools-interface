model BasicModel
	input Real u(fixed=false);
	input Real f(fixed=true);
	parameter Real scale = 3600;
	output Real x();

equation
	der(x) = (-u + f) / scale;

end BasicModel;
