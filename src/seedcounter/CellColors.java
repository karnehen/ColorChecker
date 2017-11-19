package seedcounter;

import java.util.ArrayList;
import java.util.List;

public class CellColors {
	private final List<Color> referenceColors;
	private final List<Color> actualColors;

	public CellColors() {
		referenceColors = new ArrayList<Color>();
		actualColors = new ArrayList<Color>();
	}

	public List<Color> getReferenceColors() {
		return this.referenceColors;
	}

	public List<Color> getActualColors() {
		return this.actualColors;
	}

	public void addColor(Color actualColor, Color referenceColor) {
		actualColors.add(new Color(actualColor));
		referenceColors.add(new Color(referenceColor));
	}
}
