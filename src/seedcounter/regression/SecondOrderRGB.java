package seedcounter.regression;

public class SecondOrderRGB extends AbstractRGB implements RegressionModel {
	@Override
	protected double[] bgrToFeatures(double[] bgr) {
		return new double[] {
			bgr[0], bgr[1], bgr[2],
			bgr[0] * bgr[0], bgr[0] * bgr[1], bgr[0] * bgr[2],
			bgr[1] * bgr[1], bgr[1] * bgr[2], bgr[2] * bgr[2]
		};
	}
}
