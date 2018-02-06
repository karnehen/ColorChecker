package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.List;

import seedcounter.Color;

public abstract class AbstractXYZ extends AbstractOLSMLR implements RegressionModel {
	private double[] xBeta;
	private double[] yBeta;
	private double[] zBeta;

	public AbstractXYZ(boolean intercept) {
		super(intercept);
	}

	@Override
	public void train(List<Color> train, List<Color> answers) {
		List<Double> xAnswers = new ArrayList<Double>();
		List<Double> yAnswers = new ArrayList<Double>();
		List<Double> zAnswers = new ArrayList<Double>();

		for (Color c : answers) {
			xAnswers.add(c.X());
			yAnswers.add(c.Y());
			zAnswers.add(c.Z());
		}

		xBeta = trainChannel(train, xAnswers);
		yBeta = trainChannel(train, yAnswers);
		zBeta = trainChannel(train, zAnswers);
	}

	@Override
	public Color calibrate(Color color) {
		double[] features = getFeatures(color);
		return Color.ofXYZ(new double[] {getEstimate(features, xBeta),
			getEstimate(features, yBeta), getEstimate(features, zBeta)});
	}

	@Override
	public void calibrate(DoubleBuffer color) {
		double[] features = getFeatures(Color.ofBGR(color));
		color.put(Color.ofXYZ(new double[] {getEstimate(features, xBeta),
			getEstimate(features, yBeta), getEstimate(features, zBeta)}).toBGR());
	}

	@Override
	protected double[] getFeatures(Color color) {
		return getFeatures(color.toXYZ());
	}
}
