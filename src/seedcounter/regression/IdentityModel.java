package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.List;

import seedcounter.Color;

public class IdentityModel implements RegressionModel {
	@Override
	public void train(List<Color> train, List<Color> answers) {
		;
	}

	@Override
	public Color calibrate(Color c) {
		return new Color(c);
	}

	@Override
	public void calibrate(DoubleBuffer c) {
		;
	}

}
