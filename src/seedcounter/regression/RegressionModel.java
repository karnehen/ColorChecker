package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.List;

public interface RegressionModel {
    void train(List<DoubleBuffer> train, List<DoubleBuffer> answers);
    // calculates (1 - det[H]) metric, where H is the transformation matrix from source to target features
    double getTransformationDeviance(List<DoubleBuffer> source, List<DoubleBuffer> target);
    void calibrate(DoubleBuffer c);
    String getName();
}
