package seedcounter.colormetric;

import seedcounter.Color;

public class HumanFriendlyRGB implements ColorMetric {
    public static ColorMetric create() {
        return new HumanFriendlyRGB();
    }

    @Override
    public double calculate(Color c1, Color c2) {
        double meanRed = 0.5 * (c1.red() + c2.red());
        return Math.sqrt(
            (2.0 + meanRed / 256.0) * Math.pow(c1.red() - c2.red(), 2.0) +
            4.0 * Math.pow(c1.green() - c2.green(), 2.0) +
            (2.0 + (255.0 - meanRed) / 256.0) * Math.pow(c1.blue() - c2.blue(), 2.0)
        );
    }
}
