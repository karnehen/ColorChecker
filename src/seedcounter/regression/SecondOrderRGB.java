package seedcounter.regression;

import java.nio.DoubleBuffer;

public class SecondOrderRGB extends AbstractRGB implements RegressionModel {
	@Override
	protected double[] bgrToFeatures(DoubleBuffer bgr) {
		double blue = bgr.get(bgr.position());
		double green = bgr.get(bgr.position() + 1);
		double red = bgr.get(bgr.position() + 2);
		return new double[] {
			blue, green, red,
			blue * blue, blue * green, blue * red,
			green * green, green * red, red * red
		};
	}
}
