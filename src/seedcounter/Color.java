package seedcounter;

import org.opencv.core.Scalar;

public class Color {
	private final double red;
	private final double green;
	private final double blue;

	private Color(double red, double green, double blue) {
		this.red = red;
		this.green = green;
		this.blue = blue;
	}

	public Color(Color color) {
		this(color.red, color.green, color.blue);
	}

	public double getRed() {
		return this.red;
	}

	public double getGreen() {
		return this.green;
	}

	public double getBlue() {
		return this.blue;
	}

	public static Color ofBGR(Scalar color) {
		return ofBGR(color.val);
	}

	public static Color ofBGR(double[] color) {
		return new Color(color[2], color[1], color[0]);
	}
}
