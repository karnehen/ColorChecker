package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.List;

import seedcounter.Color;

public abstract class AbstractRGB extends AbstractOLSMLR implements RegressionModel {
	private double[] redBeta;
	private double[] greenBeta;
	private double[] blueBeta;

	public AbstractRGB(boolean intercept) {
		super(intercept);
	}

	@Override
	public void train(List<Color> train, List<Color> answers) {
		List<Double> redAnswers = new ArrayList<Double>();
		List<Double> blueAnswers = new ArrayList<Double>();
		List<Double> greenAnswers = new ArrayList<Double>();

		for (Color c : answers) {
			redAnswers.add(c.red());
			blueAnswers.add(c.blue());
			greenAnswers.add(c.green());
		}

		redBeta = trainChannel(train, redAnswers);
		blueBeta = trainChannel(train, blueAnswers);
		greenBeta = trainChannel(train, greenAnswers);
	}

	@Override
	public Color calibrate(Color color) {
		double[] features = getFeatures(color.toBGR());
		return new Color(getEstimate(features, redBeta),
			getEstimate(features, greenBeta), getEstimate(features, blueBeta));
	}

	/*
	 * accepts array slice with color in BGR format
	 */
	@Override
	public void calibrate(DoubleBuffer color) {
		double[] features = getFeatures(color);
		color.put(new double[] {getEstimate(features, blueBeta),
			getEstimate(features, greenBeta), getEstimate(features, redBeta)});
	}
}
