package seedcounter;

import java.util.ArrayList;
import java.util.List;

import seedcounter.colormetric.ColorMetric;

public class CellColors {
	private final List<Color> referenceColors;
	private final List<Color> actualColors;

	public CellColors() {
		referenceColors = new ArrayList<>();
		actualColors = new ArrayList<>();
	}

	public void addColor(Color actualColor, Color referenceColor) {
		actualColors.add(new Color(actualColor));
		referenceColors.add(new Color(referenceColor));
	}

	public double calculateMetric(ColorMetric metric) {
		double sum = 0.0;
		for (int i = 0; i < referenceColors.size(); ++i) {
			sum += metric.calculate(actualColors.get(i), referenceColors.get(i));
		}
		return sum / referenceColors.size();
	}
}
