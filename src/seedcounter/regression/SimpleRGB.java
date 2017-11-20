package seedcounter.regression;

public class SimpleRGB extends AbstractRGB implements RegressionModel {
	@Override
	protected double[] bgrToFeatures(double[] bgr) {
		return new double[] {
			bgr[0], bgr[1], bgr[2]
		};
	}
}
