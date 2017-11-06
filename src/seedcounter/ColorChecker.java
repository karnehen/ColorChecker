package seedcounter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

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

	private final List<Double> xCenters = Arrays.asList(1.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0, 7.0 / 8.0);
	private final List<Double> yCenters = Arrays.asList(1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0, 6.0 / 7.0);
	private Mat image;
	private List<List<Point>> centers;
	private Integer xScale;
	private Integer yScale;

	public ColorChecker(Mat image) {
		this.image = image.clone();
		Integer width = image.width();
		Integer height = image.height();
		xScale = (int) (0.06 * width);
		yScale = (int) (0.03 * height);

		this.centers = new ArrayList<List<Point>>();
		for (Double y : yCenters) {
			List<Point> points = new ArrayList<Point>();
			for (Double x : xCenters) {
				points.add(new Point(x * width, y * height));
			}
			this.centers.add(points);
		}
	}

	public Mat brightnessCalibrationBgr(Mat srcImage) {
		Mat result = srcImage.clone();

		calibrateBgrChannel(srcImage, result, 0);
		calibrateBgrChannel(srcImage, result, 1);
		calibrateBgrChannel(srcImage, result, 2);

		return result;
	}

	private void calibrateBgrChannel(Mat srcImage, Mat dscImage, int channel) {
		List<double[]> train = new ArrayList<double[]>();
		List<Double> answers = new ArrayList<Double>();

		for (Integer row = 0; row < 6; ++row) {
			for (Integer col = 0; col < 4; ++col) {
				List<double[]> sample = getSampleColors(this.image, row, col);
				train.addAll(sample);
				for (Integer i = 0; i < sample.size(); ++i) {
					Scalar refColor = BGR_REFERENCE_COLORS.get(row).get(col);
					answers.add(refColor.val[channel]);
				}
			}
		}

		double[][] trainArray = new double[answers.size()][3];
		double[] answersArray = new double[answers.size()];
		for (int i = 0; i < answers.size(); ++i) {
			answersArray[i] = answers.get(i);
			trainArray[i] = train.get(i);
		}

		OLSMultipleLinearRegression regressor = new OLSMultipleLinearRegression();
		regressor.setNoIntercept(false);
		regressor.newSampleData(answersArray, trainArray);
		double[] beta = regressor.estimateRegressionParameters();

		for (Integer row = 0; row < srcImage.rows(); ++row) {
			for (Integer col = 0; col < srcImage.cols(); ++col) {
				double[] srcColor = srcImage.get(row, col);
				double[] dscColor = dscImage.get(row, col);
				dscColor[channel] = beta[0] + beta[1] * srcColor[0] + beta[2] * srcColor[1] + beta[3] * srcColor[2];
				dscImage.put(row, col, dscColor);
			}
		}
	}

	public Mat brightnessCalibration(Mat srcImage) {
		Mat cieImage = bgrToCie(srcImage);
		Mat referenceImage = bgrToCie(this.image);
		List<double[]> train = new ArrayList<double[]>();
		List<Double> answers = new ArrayList<Double>();

		for (Integer row = 0; row < 6; ++row) {
			Integer col = 3;
			List<double[]> sample = getSampleColors(referenceImage, row, col);
			train.addAll(sample);
			for (Integer i = 0; i < sample.size(); ++i) {
				Scalar refColor = REFERENCE_COLORS.get(row).get(col);
				answers.add(refColor.val[2]);
			}
		}

		double[][] trainArray = new double[answers.size()][3];
		double[] answersArray = new double[answers.size()];
		for (int i = 0; i < answers.size(); ++i) {
			answersArray[i] = answers.get(i);
			trainArray[i] = train.get(i);
		}

		OLSMultipleLinearRegression regressor = new OLSMultipleLinearRegression();
		regressor.setNoIntercept(true);
		regressor.newSampleData(answersArray, trainArray);
		System.out.println(regressor.isNoIntercept());
		double[] beta = regressor.estimateRegressionParameters();

		for (Integer row = 0; row < cieImage.rows(); ++row) {
			for (Integer col = 0; col < cieImage.cols(); ++col) {
				double[] color = cieImage.get(row, col);
				color[2] = beta[0] * color[0] + beta[1] * color[1] + beta[2] * color[2];
				cieImage.put(row, col, color);
			}
		}

		return cieToBgr(cieImage);
	}

	private Mat bgrToCie(Mat srcImage) {
		Mat cieImage = new Mat(srcImage.rows(), srcImage.cols(), CvType.CV_32FC3);
		Imgproc.cvtColor(srcImage, cieImage, Imgproc.COLOR_BGR2XYZ);
		Mat result = new Mat(srcImage.rows(), srcImage.cols(), CvType.CV_64FC3);


		for (Integer j = 0; j < cieImage.rows(); ++j) {
			for (Integer i = 0; i < cieImage.cols(); ++i) {
				double[] oldColor = cieImage.get(j, i);
				Double X = oldColor[0];
				Double Y = oldColor[1];
				Double Z = oldColor[2];
				double[] newColor = {X / (X + Y + Z), Y / (X + Y + Z), Y * 100.0 / 255.0};

				result.put(j, i, newColor);
			}
		}

		return result;
	}

	private Mat cieToBgr(Mat srcImage) {
		Mat result = new Mat(srcImage.rows(), srcImage.cols(), CvType.CV_32FC3);

		for (Integer j = 0; j < srcImage.rows(); ++j) {
			for (Integer i = 0; i < srcImage.cols(); ++i) {
				double[] oldColor = srcImage.get(j, i);
				Double x = oldColor[0];
				Double y = oldColor[1];
				Double Y = oldColor[2] * 255.0 / 100.0;
				double[] newColor = {x * Y / y, Y, (1.0 - x - y) * Y / y};

				result.put(j, i, newColor);
			}
		}

		Imgproc.cvtColor(result, result, Imgproc.COLOR_XYZ2BGR);

		return result;
	}

	private List<double[]> getSampleColors(Mat image, Integer row, Integer column) {
		Point center = centers.get(row).get(column);
		List<Point> points = getSurroundingPoints(center);

		List<double[]> colors = new ArrayList<double[]>();

		for (Point p : points) {
			double[] color = image.get((int)p.y, (int)p.x);
			colors.add(color);
		}

		return colors;
	}

	public Color getColor(Integer row, Integer column) {
		Point center = centers.get(row).get(column);
		List<Point> points = getSurroundingPoints(center);

		List<Double> red = new ArrayList<Double>();
		List<Double> green = new ArrayList<Double>();
		List<Double> blue = new ArrayList<Double>();
		int i = 0;

		for (Point p : points) {
			double[] color = this.image.get((int)p.y, (int)p.x);
			blue.add(color[0]);
			green.add(color[1]);
			red.add(color[2]);
			i = 255 - i;
		}

		return new Color(
			new Scalar(mean(blue), mean(green), mean(red)),
			new Scalar(std(blue), std(green), std(red))
		);
	}

	private Double mean(List<Double> numbers) {
		double result = 0.0;
		for (Double n : numbers) {
			result += n;
		}
		return result / numbers.size();
	}

	private Double std(List<Double> numbers) {
		double average = mean(numbers);
		double result = 0.0;
		for (Double n : numbers) {
			result += Math.pow(n - average, 2.0);
		}
		return Math.sqrt(result / (numbers.size() - 1));
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

	public class Color {
		private Scalar mean;
		private Scalar std;

		Color(Scalar mean, Scalar std) {
			this.mean = mean;
			this.std = std;
		}

		public Scalar mean() {
			return this.mean;
		}

		public Scalar std() {
			return this.std;
		}
	}
}
