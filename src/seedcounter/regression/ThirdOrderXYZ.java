package seedcounter.regression;

import java.nio.DoubleBuffer;

public class ThirdOrderXYZ extends AbstractXYZ implements RegressionModel {
	public ThirdOrderXYZ(boolean intercept) {
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
			y * y, y * z, z * z,
			x * x * x, x * x * y,
			x * x * z, x * y * y,
			x * y * z, x * z * z,
			y * y * y, y * y * z,
			y * z * z, z * z * z,
		};
	}
}
