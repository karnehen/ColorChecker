package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.List;

import seedcounter.Color;

public interface RegressionModel {
	public void train(List<Color> train, List<Color> answers);
	// TODO: combine the following methods
	public Color calibrate(Color c);
	public void calibrate(DoubleBuffer c);
}
