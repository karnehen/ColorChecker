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
import seedcounter.regression.RegressionModel;

public class ColorChecker {
	public final static List<List<Scalar>> BGR_REFERENCE_COLORS = Arrays.asList(
		Arrays.asList(new Scalar(171,191,99), new Scalar(41,161,229), new Scalar(166,136,0), new Scalar(50,50,50)),
		Arrays.asList(new Scalar(176,129,130), new Scalar(62,189,160), new Scalar(150,84,188), new Scalar(85,84,83)),
		Arrays.asList(new Scalar(65,108,90), new Scalar(105,59,91), new Scalar(22,200,238), new Scalar(121,121,120)),
		Arrays.asList(new Scalar(157,123,93), new Scalar(98,84,195), new Scalar(56,48,176), new Scalar(161,161,160)),
		Arrays.asList(new Scalar(129,149,196), new Scalar(168,92,72), new Scalar(72,149,71), new Scalar(201,201,200)),
		Arrays.asList(new Scalar(67,81,115), new Scalar(45,123,220), new Scalar(147,62,43), new Scalar(240,245,245))
	);

	public final static List<List<Scalar>> REFERENCE_COLORS = Arrays.asList(
			Arrays.asList(new Scalar(0.261, 0.343, 43.1), new Scalar(0.473, 0.438, 43.1), new Scalar(0.196, 0.252, 19.8), new Scalar(0.31, 0.316, 3.1)),
			Arrays.asList(new Scalar(0.265, 0.24, 24.3), new Scalar(0.38, 0.489, 44.3), new Scalar(0.364, 0.233, 19.8), new Scalar(0.31, 0.316, 9)),
			Arrays.asList(new Scalar(0.337, 0.422, 13.3), new Scalar(0.285, 0.202, 6.6), new Scalar(0.448, 0.47, 59.1), new Scalar(0.31, 0.316, 19.8)),
			Arrays.asList(new Scalar(0.247, 0.251, 19.3), new Scalar(0.453, 0.306, 19.8), new Scalar(0.539, 0.313, 12), new Scalar(0.31, 0.316, 36.2)),
			Arrays.asList(new Scalar(0.377, 0.345, 35.8), new Scalar(0.211, 0.175, 12), new Scalar(0.305, 0.478, 23.4), new Scalar(0.31, 0.316, 59.1)),
			Arrays.asList(new Scalar(0.4, 0.35, 10.1), new Scalar(0.506, 0.407, 30.1), new Scalar(0.187, 0.129, 6.1), new Scalar(0.31, 0.316, 90))
	);

	private final List<Double> xCenters = Arrays.asList(
			0.143, 0.381, 0.613, 0.862
	);
	private final List<Double> yCenters = Arrays.asList(
			0.160, 0.305, 0.440, 0.580, 0.717, 0.856
	);
	private Mat checkerImage;
	private List<List<Point>> centers;
	private Integer xScale;
	private Integer yScale;

	private static final Double REAL_WIDTH = 64.0; // millimeters
	private static final Double REAL_HEIGHT = 108.0; // millimeters

	public ColorChecker(Mat image) {
		this.checkerImage = image.clone();
		Integer width = image.width();
		Integer height = image.height();
		xScale = (int) (0.04 * width);
		yScale = (int) (0.02 * height);

		this.centers = new ArrayList<List<Point>>();
		for (Double y : yCenters) {
			List<Point> points = new ArrayList<Point>();
			for (Double x : xCenters) {
				points.add(new Point(x * width, y * height));
			}
			this.centers.add(points);
		}
	}

	public Double pixelArea(Quad quad) {
		return REAL_WIDTH * REAL_HEIGHT / quad.getArea();
	}

	public Mat calibrationBgr(Mat srcImage, RegressionModel model) {
		Mat result = srcImage.clone();
		List<Color> train = new ArrayList<Color>();
		List<Color> answers = new ArrayList<Color>();

		for (Integer row = 0; row < 6; ++row) {
			for (Integer col = 0; col < 4; ++col) {
				List<double[]> sample = getSampleColors(this.checkerImage, row, col, false);
				for (Integer i = 0; i < sample.size(); ++i) {
					train.add(Color.ofBGR(sample.get(i)));
					answers.add(Color.ofBGR(BGR_REFERENCE_COLORS.get(row).get(col)));
				}
			}
		}

		try {
			model.train(train, answers);
		} catch (SingularMatrixException e) {
			return result;
		}

		result.convertTo(result, CvType.CV_64FC3);
		int channels = result.channels();
		int size = (int) result.total() * channels;
		double[] temp = new double[size];
		result.get(0, 0, temp);
		for (int i = 0; i + channels < size; i += channels) {
			DoubleBuffer srcColor = DoubleBuffer.wrap(temp, i, channels);
			model.calibrate(srcColor);
		}
		result.put(0, 0, temp);
		result.convertTo(result, srcImage.type());

		return result;
	}

	public CellColors getCellColors(Mat checkerImage) {
		CellColors cellColors = new CellColors();

		for (Integer row = 0; row < 6; ++row) {
			for (Integer col = 0; col < 4; ++col) {
				List<double[]> actualColors = getSampleColors(checkerImage, row, col, true);
				Color referenceColor = Color.ofBGR(BGR_REFERENCE_COLORS.get(row).get(col));
				for (double[] c : actualColors) {
					cellColors.addColor(Color.ofBGR(c), referenceColor);
				}
			}
		}

		return cellColors;
	}

	private List<double[]> getSampleColors(Mat checkerImage, Integer row,
			Integer col, boolean allPoints) {
		final int STEP = 10;
		Point center = centers.get(row).get(col);
		List<Point> points = getSurroundingPoints(center);

		List<double[]> colors = new ArrayList<double[]>();

		if (allPoints) {
			int minX = (int) points.get(0).x;
			int minY = (int) points.get(0).y;
			int maxX = (int) points.get(8).x;
			int maxY = (int) points.get(8).y;
			for (int y = minY; y <= maxY; y += STEP) {
				for (int x = minX; x <= maxX; x += STEP) {
					double[] color = checkerImage.get(y, x);
					colors.add(color);
				}
			}
		} else {
			for (Point p : points) {
				double[] color = checkerImage.get((int)p.y, (int)p.x);
				colors.add(color);
			}
		}

		return colors;
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
						Core.circle(result, p, 10, red, Core.FILLED);
					} else {
						Core.circle(result, p, 10, blue, Core.FILLED);
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
}
