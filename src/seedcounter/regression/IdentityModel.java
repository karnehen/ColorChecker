package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.List;

import seedcounter.Color;
import seedcounter.colormetric.ColorMetric;
import seedcounter.colormetric.EuclideanRGB;

public class IdentityModel implements RegressionModel {
	private double aic;

	@Override
	public void train(List<Color> train, List<Color> answers) {
		int samples = train.size();
		double rss = 0.0;
		ColorMetric metric = EuclideanRGB.create();

		for (int i = 0; i < samples; ++i) {
			rss += metric.calculate(train.get(i), answers.get(i));
		}

		aic = samples / Math.log(rss / samples);
	}

	@Override
	public void calibrate(DoubleBuffer c) {}

	@Override
	public double getAIC() {
		return aic;
	}
}
