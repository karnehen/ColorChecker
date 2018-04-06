package seedcounter.colormetric;

import seedcounter.Color;

public class EuclideanRGB implements ColorMetric {
    public static ColorMetric create() {
        return new EuclideanRGB();
    }

    @Override
    public double calculate(Color c1, Color c2) {
        return Math.sqrt(
            Math.pow(c1.red() - c2.red(), 2.0) +
            Math.pow(c1.green() - c2.green(), 2.0) +
            Math.pow(c1.blue() - c2.blue(), 2.0)
        );
    }
}
