package seedcounter.regression;

import seedcounter.colormetric.Color;

import java.nio.DoubleBuffer;

public class SimpleOLS extends AbstractOLS implements RegressionModel {
    public SimpleOLS(boolean intercept) {
        super(intercept);
    }

    @Override
    protected double[] getFeatures(DoubleBuffer color) {
        double channel0 = Color.channel(color, 0);
        double channel1 = Color.channel(color, 1);
        double channel2 = Color.channel(color, 2);

        return new double[] {
            channel0, channel1, channel2
        };
    }
}
