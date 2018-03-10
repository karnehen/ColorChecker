package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.List;
import java.util.stream.DoubleStream;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import seedcounter.Color;

public abstract class AbstractOLSMLR implements RegressionModel {
	private final boolean intercept;

	AbstractOLSMLR(boolean intercept) {
		this.intercept = intercept;
	}

	RegressionResult trainChannel(List<Color> trainSet, List<Double> answers) {
		double[][] trainArray = new double[answers.size()][3];
		double[] answersArray = new double[answers.size()];

		for (int i = 0; i < answers.size(); ++i) {
			answersArray[i] = answers.get(i);
			trainArray[i] = getFeatures(trainSet.get(i));
		}

		OLSMultipleLinearRegression regressor = new OLSMultipleLinearRegression();
		regressor.setNoIntercept(!intercept);
		regressor.newSampleData(answersArray, trainArray);

		return new RegressionResult(regressor);
	}

	double getEstimate(double[] features, double[] beta) {
		double answer = (intercept ? 1 : 0) * beta[0];
		for (int i = 0; i < features.length; ++i) {
			answer += features[i] * beta[i + (intercept ? 1 : 0)];
		}

		return answer;
	}

	@Override
	abstract public void train(List<Color> train, List<Color> answers);

	@Override
	abstract public void calibrate(DoubleBuffer color);

	abstract protected double[] getFeatures(Color color);

	abstract protected double[] getFeatures(DoubleBuffer color);

	public class RegressionResult {
		public RegressionResult(OLSMultipleLinearRegression regressor) {
			this.features = regressor.estimateRegressionParameters();
		    int samples = regressor.estimateResiduals().length;
			double rss = regressor.calculateResidualSumOfSquares();
			this.aic = 2.0 * this.features.length + samples * Math.log(rss / samples);
		}

		private final double[] features;
		private final double aic;

		public double[] getFeatures() {
			return features;
		}

		public double getAIC() {
			return aic;
		}
	}
}
