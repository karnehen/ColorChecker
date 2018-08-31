package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.List;

public class IdentityModel implements RegressionModel {
    @Override
    public void train(List<DoubleBuffer> train, List<DoubleBuffer> answers) {}

    @Override
    public double getTransformationDeviance(List<DoubleBuffer> source, List<DoubleBuffer> target) {
        return 0.0;
    }

    @Override
    public void calibrate(DoubleBuffer c) {}

    @Override
    public String getName() {
        return "Identity";
    }
}
