package seedcounter.regression;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

import seedcounter.colormetric.Color;

public abstract class AbstractOLS implements RegressionModel {
    private final boolean intercept;
    private double[] beta1;
    private double[] beta2;
    private double[] beta3;

    AbstractOLS(boolean intercept) {
        this.intercept = intercept;
    }

    @Override
    public void train(List<DoubleBuffer> train, List<DoubleBuffer> answers) {
        List<Double> answers1 = new ArrayList<>();
        List<Double> answers2 = new ArrayList<>();
        List<Double> answers3 = new ArrayList<>();

        for (DoubleBuffer c : answers) {
            answers1.add(Color.channel(c, 0));
            answers2.add(Color.channel(c, 1));
            answers3.add(Color.channel(c, 2));
        }

        beta1 = trainChannel(train, answers1);
        beta2 = trainChannel(train, answers2);
        beta3 = trainChannel(train, answers3);
    }

    @Override
    public double getTransformationDeviance(List<DoubleBuffer> source, List<DoubleBuffer> target) {
        List<List<Double>> targetFeatures = new ArrayList<>();

        int featuresCount = getFeatures(target.get(0)).length;

        for (int i = 0; i < featuresCount; ++i) {
            targetFeatures.add(new ArrayList<>());
        }

        for (DoubleBuffer c : target) {
            double[] features = getFeatures(c);

            for (int i = 0; i < features.length; ++i) {
                targetFeatures.get(i).add(features[i]);
            }
        }

        int dim = featuresCount + (intercept ? 1 : 0);
        RealMatrix matrix = new Array2DRowRealMatrix(dim, dim);

        if (intercept) {
            matrix.setEntry(0, 0, 1.0);
            for (int col = 1; col < dim; ++col) {
                matrix.setEntry(0, col, 0.0);
            }
        }

        for (int index = 0; index < targetFeatures.size(); ++index) {
            List<Double> answers = targetFeatures.get(index);
            int row = index + (intercept ? 1 : 0);

            matrix.setRow(row, trainChannel(source, answers));
        }

        return 1.0 - new LUDecomposition(matrix).getDeterminant();
    }

    private double[] trainChannel(List<DoubleBuffer> trainSet, List<Double> answers) {
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

    abstract protected double[] getFeatures(DoubleBuffer color);
}
