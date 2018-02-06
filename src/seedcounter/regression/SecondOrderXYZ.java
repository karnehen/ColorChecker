package seedcounter.regression;

import java.nio.DoubleBuffer;

public class SecondOrderXYZ extends AbstractXYZ implements RegressionModel {
	public SecondOrderXYZ(boolean intercept) {
		super(intercept);
	}

	@Override
	protected double[] getFeatures(DoubleBuffer xyz) {
		double x = xyz.get(xyz.position());
		double y = xyz.get(xyz.position() + 1);
		double z = xyz.get(xyz.position() + 2);
		return new double[] {
			x, y, z,
			x * x, x * y, x * z,
			y * y, y * z, z * z
		};
	}
}
