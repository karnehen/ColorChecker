package seedcounter.colormetric;

import seedcounter.Color;

public class EuclideanLab implements ColorMetric {
    public static EuclideanLab create() {
        return new EuclideanLab();
    }

    @Override
    public double calculate(Color c1, Color c2) {
        return Math.sqrt(
                Math.pow(c1.lightness() - c2.lightness(), 2.0) +
                Math.pow(c1.a() - c2.a(), 2.0) +
                Math.pow(c1.b() - c2.b(), 2.0)
            );
    }

}
