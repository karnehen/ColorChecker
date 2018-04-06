package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import seedcounter.Color;

public abstract class AbstractOLS implements RegressionModel {
    private final boolean intercept;
    private double[] beta1;
    private double[] beta2;
    private double[] beta3;

    AbstractOLS(boolean intercept) {
        this.intercept = intercept;
    }

    public void train(List<Color> train, List<Color> answers) {
        List<Double> answers1 = new ArrayList<>();
        List<Double> answers2 = new ArrayList<>();
        List<Double> answers3 = new ArrayList<>();

        for (Color c : answers) {
            answers1.add(c.blue());
            answers2.add(c.green());
            answers3.add(c.red());
        }

        beta1 = trainChannel(train, answers1);
        beta2 = trainChannel(train, answers2);
        beta3 = trainChannel(train, answers3);
    }

    private double[] trainChannel(List<Color> trainSet, List<Double> answers) {
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

    private double getEstimate(double[] features, double[] beta) {
        double answer = (intercept ? 1 : 0) * beta[0];
        for (int i = 0; i < features.length; ++i) {
            answer += features[i] * beta[i + (intercept ? 1 : 0)];
        }

        return answer;
    }

    @Override
    public void calibrate(DoubleBuffer color) {
        double[] features = getFeatures(color);
        color.put(new double[] {getEstimate(features, beta1),
                getEstimate(features, beta2), getEstimate(features, beta3)});
    }

    @Override
    public String getName() {
        return this.getClass().getSimpleName() + (intercept ? "Intercept" : "");
    }

    private double[] getFeatures(Color color) {
        return getFeatures(color.toBGR());
    }

    abstract protected double[] getFeatures(DoubleBuffer color);
}
