package seedcounter;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import seedcounter.regression.RegressionModel;

public class ColorChecker {
	private final static List<List<Scalar>> BGR_REFERENCE_COLORS = Arrays.asList(
			Arrays.asList(new Scalar(171, 191, 99), new Scalar(41, 161, 229), new Scalar(166, 136, 0), new Scalar(50, 50, 50)),
			Arrays.asList(new Scalar(176, 129, 130), new Scalar(62, 189, 160), new Scalar(150, 84, 188), new Scalar(85, 84, 83)),
			Arrays.asList(new Scalar(65, 108, 90), new Scalar(105, 59, 91), new Scalar(22, 200, 238), new Scalar(121, 121, 120)),
			Arrays.asList(new Scalar(157, 123, 93), new Scalar(98, 84, 195), new Scalar(56, 48, 176), new Scalar(161, 161, 160)),
			Arrays.asList(new Scalar(129, 149, 196), new Scalar(168, 92, 72), new Scalar(72, 149, 71), new Scalar(201, 201, 200)),
			Arrays.asList(new Scalar(67, 81, 115), new Scalar(45, 123, 220), new Scalar(147, 62, 43), new Scalar(240, 245, 245))
	);

	private final Mat checkerImage;
	private final List<List<Point>> centers;
	private final Integer xScale;
	private final Integer yScale;

	private static final Double REAL_WIDTH = 64.0; // millimeters
	private static final Double REAL_HEIGHT = 108.0; // millimeters

	public ColorChecker(Mat image) {
		this.checkerImage = image.clone();
		Integer width = image.width();
		Integer height = image.height();

		xScale = (int) (0.04 * width);
		yScale = (int) (0.02 * height);
		List<Double> xCenters = Arrays.asList(0.143, 0.381, 0.613, 0.862);
		List<Double> yCenters = Arrays.asList(0.160, 0.305, 0.440, 0.580, 0.717, 0.856);

		this.centers = new ArrayList<>();
		for (Double y : yCenters) {
			List<Point> points = new ArrayList<>();
			for (Double x : xCenters) {
				points.add(new Point(x * width, y * height));
			}
			this.centers.add(points);
		}
	}

	public Double pixelArea(Quad quad) {
		return REAL_WIDTH * REAL_HEIGHT / quad.getArea();
	}

	public Mat calibrate(Mat srcImage, RegressionModel model,
						 ColorSpace featuresSpace, ColorSpace targetSpace) {
		Mat result = srcImage.clone();
		result.convertTo(result, CvType.CV_64FC3);
		List<Color> train = new ArrayList<>();
		List<Color> answers = new ArrayList<>();

		int channels = result.channels();
		int size = (int) result.total() * channels;
		double[] temp = new double[size];
		result.get(0, 0, temp);

		for (int i = 0; i + channels < size; i += channels) {
			double b = temp[i];
			double g = temp[i + 1];
			double r = temp[i + 2];

			if (featuresSpace.isLinear()) {
				b = Color.linearizeRGB(b);
				g = Color.linearizeRGB(g);
				r = Color.linearizeRGB(r);
			}

			if (featuresSpace.isXYZ()) {
				temp[i] = r * 0.4124
						+ g * 0.3576
						+ b * 0.1805;
				temp[i + 1] = r * 0.2126
						+ g * 0.7152
						+ b * 0.0722;
				temp[i + 2] = r * 0.0193
						+ g * 0.1192
						+ b * 0.9505;
			} else {
				temp[i] = b;
				temp[i + 1] = g;
				temp[i + 2] = r;
			}
		}

		for (Integer row = 0; row < BGR_REFERENCE_COLORS.size(); ++row) {
			for (Integer col = 0; col < BGR_REFERENCE_COLORS.get(0).size(); ++col) {
				List<ColoredPoint> samplePoints = getSamplePoints(row, col, false);
				for (ColoredPoint s : samplePoints) {
					train.add(Color.ofBGR(ConvertColors(s.b, s.g, s.r, featuresSpace)));
					double[] referenceColor = BGR_REFERENCE_COLORS.get(row).get(col).val;
					answers.add(Color.ofBGR(ConvertColors(referenceColor[0], referenceColor[1],
							referenceColor[2], targetSpace)));
				}
			}
		}

		try {
			model.train(train, answers);
		} catch (SingularMatrixException e) {
			return result;
		}

		for (int i = 0; i + channels < size; i += channels) {
			DoubleBuffer srcColor = DoubleBuffer.wrap(temp, i, channels);
			model.calibrate(srcColor);
		}

		for (int i = 0; i + channels < size; i += channels) {
			if (targetSpace.isXYZ()) {
				double x = temp[i];
				double y = temp[i + 1];
				double z = temp[i + 2];
				temp[i] = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
				temp[i + 1] = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
				temp[i + 2] = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
			}
			if (targetSpace.isLinear()) {
				temp[i] = Color.inverseLinearizeRGB(temp[i]);
				temp[i + 1] = Color.inverseLinearizeRGB(temp[i + 1]);
				temp[i + 2] = Color.inverseLinearizeRGB(temp[i + 2]);
			}
		}

		result.put(0, 0, temp);
		result.convertTo(result, srcImage.type());

		return result;
	}

	public CellColors getCellColors(Mat checkerImage) {
		CellColors cellColors = new CellColors();

		for (Integer row = 0; row < 6; ++row) {
			for (Integer col = 0; col < 4; ++col) {
				List<ColoredPoint> actualColors = getSamplePoints(row, col, true);
				Color referenceColor = Color.ofBGR(BGR_REFERENCE_COLORS.get(row).get(col));
				for (ColoredPoint c : actualColors) {
					cellColors.addColor(Color.ofBGR(new double[]{c.b, c.r, c.g}), referenceColor);
				}
			}
		}

		return cellColors;
	}

	private List<ColoredPoint> getSamplePoints(Integer row, Integer col, boolean allPoints) {
		final int STEP = 10;
		Point center = centers.get(row).get(col);
		List<Point> surroundingPoints = getSurroundingPoints(center);

		List<ColoredPoint> points = new ArrayList<>();

		if (allPoints) {
			int minX = (int) surroundingPoints.get(0).x;
			int minY = (int) surroundingPoints.get(0).y;
			int maxX = (int) surroundingPoints.get(8).x;
			int maxY = (int) surroundingPoints.get(8).y;
			for (int y = minY; y <= maxY; y += STEP) {
				for (int x = minX; x <= maxX; x += STEP) {
					double[] c = this.checkerImage.get(y, x);
					points.add(new ColoredPoint(x, y, c[0], c[1], c[2]));
				}
			}
		} else {
			for (Point p : surroundingPoints) {
				double[] c = this.checkerImage.get((int) p.y, (int) p.x);
				points.add(new ColoredPoint((int) p.x, (int) p.y, c[0], c[1], c[2]));
			}
		}

		return points;
	}

	public Mat drawSamplePoints() {
		Mat result = checkerImage.clone();
		Scalar red = new Scalar(0, 0, 255);
		Scalar blue = new Scalar(255, 0, 0);

		for (Integer row = 0; row < 6; ++row) {
			for (Integer col = 0; col < 4; ++col) {
				Point center = centers.get(row).get(col);
				List<Point> points = getSurroundingPoints(center);
				int i = 0;
				for (Point p : points) {
					if (i % 2 == 0) {
						Imgproc.circle(result, p, 10, red, Core.FILLED);
					} else {
						Imgproc.circle(result, p, 10, blue, Core.FILLED);
					}
					i += 1;
				}
			}
		}

		return result;
	}

	private List<Point> getSurroundingPoints(Point center) {
		return Arrays.asList(
				new Point(center.x - this.xScale, center.y - this.yScale),
				new Point(center.x, center.y - this.yScale),
				new Point(center.x + this.xScale, center.y - this.yScale),
				new Point(center.x - this.xScale, center.y),
				new Point(center.x, center.y),
				new Point(center.x + this.xScale, center.y),
				new Point(center.x - this.xScale, center.y + this.yScale),
				new Point(center.x, center.y + this.yScale),
				new Point(center.x + this.xScale, center.y + this.yScale)
		);
	}

	private class ColoredPoint {
		public final double b;
		public final double g;
		public final double r;
		public final int x;
		public final int y;

		ColoredPoint(int x, int y, double b, double g, double r) {
			this.x = x;
			this.y = y;
			this.b = b;
			this.g = g;
			this.r = r;
		}
	}

	private double[] ConvertColors(double b, double g, double r, ColorSpace space) {
		if (space.isLinear()) {
			b = Color.linearizeRGB(b);
			g = Color.linearizeRGB(g);
			r = Color.linearizeRGB(r);
		}
		double[] result = {0.0, 0.0, 0.0};

		if (space.isXYZ()) {
			result[0] = r * 0.4124
					+ g * 0.3576
					+ b * 0.1805;
			result[1] = r * 0.2126
					+ g * 0.7152
					+ b * 0.0722;
			result[2] = r * 0.0193
					+ g * 0.1192
					+ b * 0.9505;
		} else {
			result[0] = b;
			result[1] = g;
			result[2] = r;
		}

		return result;
	}
}