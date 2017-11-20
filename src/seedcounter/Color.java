package seedcounter;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class Color {
	private final double red;
	private final double green;
	private final double blue;
	private final double lightness;
	private final double a;
	private final double b;

	public Color(double red, double green, double blue) {
		this.red = red;
		this.green = green;
		this.blue = blue;

		Scalar lab = bgrToLabScalar(new Scalar(blue, green, red));
		this.lightness = lab.val[0];
		this.a = lab.val[1];
		this.b = lab.val[2];
	}

	public Color(Color color) {
		this(color.red, color.green, color.blue);
	}

	public double red() {
		return this.red;
	}

	public double green() {
		return this.green;
	}

	public double blue() {
		return this.blue;
	}

	public double lightness() {
		return this.lightness;
	}

	public double a() {
		return this.a;
	}

	public double b() {
		return this.b;
	}

	public double[] toBGR() {
		return new Scalar(this.blue, this.green, this.red).val;
	}

	public static Color ofBGR(Scalar color) {
		return ofBGR(color.val);
	}

	public static Color ofBGR(double[] color) {
		return new Color(color[2], color[1], color[0]);
	}

	private Scalar bgrToLabScalar(Scalar color) {
		Mat bgr = new Mat(1, 1, CvType.CV_32FC3, color);
		Mat lab = new Mat(1, 1, CvType.CV_32FC3);
		Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab);

		return new Scalar(lab.get(0,  0));
	}
}
