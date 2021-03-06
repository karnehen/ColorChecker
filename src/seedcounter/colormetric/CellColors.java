package seedcounter.colormetric;

import java.util.ArrayList;
import java.util.List;

import seedcounter.colormetric.Color;
import seedcounter.colormetric.ColorMetric;

public class CellColors {
    private final List<Color> referenceColors;
    private final List<Color> actualColors;

    public CellColors() {
        referenceColors = new ArrayList<>();
        actualColors = new ArrayList<>();
    }

    public void addColor(Color actualColor, Color referenceColor) {
        actualColors.add(actualColor);
        referenceColors.add(referenceColor);
    }

    public double calculateMetric(ColorMetric metric) {
        double sum = 0.0;
        for (int i = 0; i < referenceColors.size(); ++i) {
            sum += metric.calculate(actualColors.get(i), referenceColors.get(i));
        }
        return sum / referenceColors.size();
    }
}
