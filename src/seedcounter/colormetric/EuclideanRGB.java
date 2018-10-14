package seedcounter.colormetric;

public class EuclideanRGB implements ColorMetric {
    @Override
    public double calculate(Color c1, Color c2) {
        return Math.sqrt(
            Math.pow(c1.red() - c2.red(), 2.0) +
            Math.pow(c1.green() - c2.green(), 2.0) +
            Math.pow(c1.blue() - c2.blue(), 2.0)
        );
    }
}
