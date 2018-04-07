package seedcounter.regression;

import seedcounter.colormetric.Color;

import java.nio.DoubleBuffer;

public class SecondOrderOLS extends AbstractOLS implements RegressionModel {
    public SecondOrderOLS(boolean intercept) {
        super(intercept);
    }

    @Override
    protected double[] getFeatures(DoubleBuffer color) {
        double channel0 = Color.channel(color, 0);
        double channel1 = Color.channel(color, 1);
        double channel2 = Color.channel(color, 2);

        return new double[] {
            channel0, channel1, channel2,
            channel0 * channel0, channel0 * channel1, channel0 * channel2,
            channel1 * channel1, channel1 * channel2, channel2 * channel2
        };
    }
}
