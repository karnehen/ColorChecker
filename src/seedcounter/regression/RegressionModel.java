package seedcounter.regression;

import java.util.List;

import seedcounter.Color;

public interface RegressionModel {
	public void train(List<Color> train, List<Color> answers);
	public Color calibrate(Color c);
}
