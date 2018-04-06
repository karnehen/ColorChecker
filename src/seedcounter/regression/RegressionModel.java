package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.List;

import seedcounter.Color;

public interface RegressionModel {
    void train(List<DoubleBuffer> train, List<DoubleBuffer> answers);
    void calibrate(DoubleBuffer c);
    String getName();
}
