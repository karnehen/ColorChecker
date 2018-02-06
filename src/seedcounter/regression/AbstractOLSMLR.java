package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.List;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import seedcounter.Color;

public abstract class AbstractOLSMLR implements RegressionModel {
	private final boolean intercept;

	public AbstractOLSMLR(boolean intercept) {
		this.intercept = intercept;
	}

	protected double[] trainChannel(List<Color> trainSet, List<Double> answers) {
		double[][] trainArray = new double[answers.size()][3];
		double[] answersArray = new double[answers.size()];

		for (int i = 0; i < answers.size(); ++i) {
			answersArray[i] = answers.get(i);
			trainArray[i] = getFeatures(trainSet.get(i));
		}

		OLSMultipleLinearRegression regressor = new OLSMultipleLinearRegression();
		regressor.setNoIntercept(!intercept);
		regressor.newSampleData(answersArray, trainArray);

		return regressor.estimateRegressionParameters();
	}

	protected double getEstimate(double[] features, double[] beta) {
		double answer = (intercept ? 1 : 0) * beta[0];
		for (int i = 0; i < features.length; ++i) {
			answer += features[i] * beta[i + (intercept ? 1 : 0)];
		}

		return answer;
	}

	@Override
	abstract public void train(List<Color> train, List<Color> answers);

	@Override
	abstract public Color calibrate(Color color);

	@Override
	abstract public void calibrate(DoubleBuffer color);

	abstract protected double[] getFeatures(Color color);

	abstract protected double[] getFeatures(DoubleBuffer color);
}
