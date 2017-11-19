package seedcounter.colormetric;

import seedcounter.Color;

public class HumanFriendlyRGB implements ColorMetric {
	public static ColorMetric create() {
		return new HumanFriendlyRGB();
	}

	@Override
	public double calculate(Color c1, Color c2) {
		double meanRed = 0.5 * (c1.getRed() + c2.getRed());
		return Math.sqrt(
			(2.0 + meanRed / 256.0) * Math.pow(c1.getRed() - c2.getRed(), 2.0) +
			4.0 * Math.pow(c1.getGreen() - c2.getGreen(), 2.0) +
			(2.0 + (255.0 - meanRed) / 256.0) * Math.pow(c1.getBlue() - c2.getBlue(), 2.0)
		);
	}
}
