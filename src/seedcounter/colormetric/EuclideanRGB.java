package seedcounter.colormetric;

import seedcounter.Color;

public class EuclideanRGB implements ColorMetric {
	public static ColorMetric create() {
		return new EuclideanRGB();
	}

	@Override
	public double calculate(Color c1, Color c2) {
		return Math.sqrt(
			Math.pow(c1.getRed() - c2.getRed(), 2.0) +
			Math.pow(c1.getGreen() - c2.getGreen(), 2.0) +
			Math.pow(c1.getBlue() - c2.getBlue(), 2.0)
		);
	}
}
