package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import seedcounter.Color;

public abstract class AbstractRGB implements RegressionModel {
	private double[] redBeta;
	private double[] greenBeta;
	private double[] blueBeta;

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
		double[] features = bgrToFeatures(color.toBGR());
		return new Color(getEstimate(features, redBeta),
			getEstimate(features, greenBeta), getEstimate(features, blueBeta));
	}

	/*
	 * accepts array slice with color in BGR format
	 */
	@Override
	public void calibrate(DoubleBuffer color) {
		double[] features = bgrToFeatures(color);
		color.put(new double[] {getEstimate(features, blueBeta),
			getEstimate(features, greenBeta), getEstimate(features, redBeta)});
	}

	private double[] trainChannel(List<Color> trainSet, List<Double> answers) {
		double[][] trainArray = new double[answers.size()][3];
		double[] answersArray = new double[answers.size()];

		for (int i = 0; i < answers.size(); ++i) {
			answersArray[i] = answers.get(i);
			trainArray[i] = bgrToFeatures(trainSet.get(i).toBGR());
		}

		OLSMultipleLinearRegression regressor = new OLSMultipleLinearRegression();
		regressor.setNoIntercept(false);
		regressor.newSampleData(answersArray, trainArray);

		return regressor.estimateRegressionParameters();
	}

	abstract protected double[] bgrToFeatures(DoubleBuffer bgr);

	// TODO: remove
	private double[] bgrToFeatures(double[] bgr) {
		return bgrToFeatures(DoubleBuffer.wrap(bgr));
	}

	private double getEstimate(double[] features, double[] beta) {
		double answer = beta[0];
		for (int i = 0; i < features.length; ++i) {
			answer += features[i] * beta[i+1];
		}

		return answer;
	}
}
