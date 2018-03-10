package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.List;

import seedcounter.Color;

public interface RegressionModel {
	void train(List<Color> train, List<Color> answers);
	void calibrate(DoubleBuffer c);
	double getAIC();
}
