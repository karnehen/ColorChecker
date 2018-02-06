package seedcounter.regression;

import java.nio.DoubleBuffer;

public class ThirdOrderRGB extends AbstractRGB implements RegressionModel {
	public ThirdOrderRGB(boolean intercept) {
		super(intercept);
	}

	@Override
	protected double[] getFeatures(DoubleBuffer bgr) {
		double blue = bgr.get(bgr.position());
		double green = bgr.get(bgr.position() + 1);
		double red = bgr.get(bgr.position() + 2);
		return new double[] {
			blue, green, red,
			blue * blue, blue * green, blue * red,
			green * green, green * red, red * red,
			blue * blue * blue, blue * blue * green,
			blue * blue * red, blue * green * green,
			blue * green * red, blue * red * red,
			green * green * green, green * green * red,
			green * red * red, red * red * red,
		};
	}
}
