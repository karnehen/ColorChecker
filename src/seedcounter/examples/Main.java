package seedcounter.examples;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;

import javafx.util.Pair;

import javax.activation.MimetypesFileTypeMap;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.opencv.xfeatures2d.SIFT;
import org.opencv.xfeatures2d.SURF;
import seedcounter.ColorChecker;
import seedcounter.FindColorChecker;
import seedcounter.Helper;
import seedcounter.MatchingModel;
import seedcounter.Quad;
import seedcounter.colormetric.ColorMetric;
import seedcounter.colormetric.EuclideanLab;
import seedcounter.colormetric.EuclideanRGB;
import seedcounter.colormetric.HumanFriendlyRGB;
import seedcounter.regression.RegressionFactory;
import seedcounter.ColorSpace;
import seedcounter.regression.RegressionFactory.Order;
import seedcounter.regression.RegressionModel;

class Main {
	private static final String INPUT_DIRECTORY = "../../photos/SPH-L900";
	private static final String REFERENCE_FILE = "reference.png";
	// targets and ranges
	private static final List<Pair<Scalar, Scalar>> SEED_TYPES = Arrays.asList(
			new Pair<>(new Scalar(4, 97, 108), new Scalar(50, 100, 80)),
			new Pair<>(new Scalar(17, 67, 232), new Scalar(50, 50, 50))
		);

	private static void printMap(PrintWriter writer, Map<String, String> map) {
		boolean header = map.containsKey("header");
		map.remove("header");
		StringBuilder builder = new StringBuilder();
		if (header) {
			for (String key : map.keySet()) {
				builder.append(key);
				builder.append("\t");
			}
			builder.deleteCharAt(builder.length() - 1);
			builder.append("\n");
		}
		for (String key : map.keySet()) {
			builder.append(map.get(key));
			builder.append("\t");
		}
		builder.deleteCharAt(builder.length() - 1);
		writer.println(builder.toString());
	}

	private static void printSingleSeed(MatOfPoint contour, Mat image, Mat seedBuffer,
			PrintWriter writer, Map<String, String> data) {
		Imgproc.drawContours(seedBuffer, Collections.singletonList(contour), 0,
				new Scalar(255.0), Core.FILLED);

		int minX = image.cols() - 1;
		int maxX = 0;
		int minY = image.rows() - 1;
		int maxY = 0;
		
		for (Point point : contour.toList()) {
			if (point.x < minX) {
				minX = (int) point.x;
			}
			if (point.x > maxX) {
				maxX = (int) point.x;
			}
			if (point.y < minY) {
				minY = (int) point.y;
			}
			if (point.y > maxY) {
				maxY = (int) point.y;
			}
		}

		int counter = 0;

		for (int y = minY; y <= maxY; ++y) {
			for (int x = minX; x <= maxX; ++x) {
				if (seedBuffer.get(y, x)[0] > 0.0) {
					double[] color = image.get(y, x);
					if (color[0] + color[1] + color[2] > 0.0) {
						if (counter % 10 == 0) {
							data.put("x", String.valueOf(x));
							data.put("y", String.valueOf(y));
							data.put("blue", String.valueOf(color[0]));
							data.put("green", String.valueOf(color[1]));
							data.put("red", String.valueOf(color[2]));
							printMap(writer, data);
						}
						counter += 1;
					}
				}
			}
		}
		Imgproc.drawContours(seedBuffer, Collections.singletonList(contour), 0, new Scalar(0.0));
	}

	private static void printSeeds(Mat image, PrintWriter writer,
			Map<String, String> data, Double scale) {
		List<MatOfPoint> contours = Helper.getContours(image);
		Mat seedBuffer = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);

		for (int i = 0; i < contours.size(); ++i) {
			data.put("seed_number", String.valueOf(i));
			MatOfPoint contour = contours.get(i);
			Double area = scale * Imgproc.contourArea(contour);
			if (area < 50.0) {
				data.put("area", area.toString());
				printSingleSeed(contours.get(i), image, seedBuffer, writer, data);
			}
			contour.release();
		}

		seedBuffer.release();
	}

	private static Mat filterByMask(Mat image, Mat mask) {
		Mat filtered = Helper.filterByMask(image, mask);
		Range rows = new Range(filtered.rows() / 4, 3 * filtered.rows() / 4);
		Range cols = new Range(filtered.cols() / 4, 3 * filtered.cols() / 4);
		return new Mat(filtered, rows, cols);
	}

	private static Mat getMask(Mat image) {
		Mat mask = Helper.binarizeSeed(image, SEED_TYPES);
		Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(50, 50));
		Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
	
		Mat whiteMask = Helper.whiteThreshold(image);
		kernel.release();
		kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(10, 10));
		Imgproc.morphologyEx(whiteMask, whiteMask, Imgproc.MORPH_CLOSE, kernel);
	
		Core.bitwise_and(mask, whiteMask, mask);
		kernel.release();
	
		return mask;
	}

	public static void main(String[] args) throws FileNotFoundException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		 List<MatchingModel> MATCHING_MODELS = Arrays.asList(
				new MatchingModel(SURF.create(), SURF.create(),
						DescriptorMatcher.FLANNBASED, 0.7f),
				new MatchingModel(SIFT.create(), SIFT.create(),
						DescriptorMatcher.FLANNBASED, 0.7f),
				new MatchingModel(ORB.create(), ORB.create(),
						DescriptorMatcher.BRUTEFORCE_HAMMING, 0.9f)
		);
		FindColorChecker f = new FindColorChecker(REFERENCE_FILE, MATCHING_MODELS.get(1));

		File inputDirectory = new File(INPUT_DIRECTORY);
		new File(inputDirectory.getAbsolutePath() + "/result").mkdir();

		List<RegressionModel> models = new ArrayList<>();
		models.add(RegressionFactory.createModel(Order.FIRST));
		models.add(RegressionFactory.createModel(Order.SECOND));
		models.add(RegressionFactory.createModel(Order.THIRD));
		models.add(RegressionFactory.createModel(Order.IDENTITY));

		List<ColorMetric> metrics = new ArrayList<>();
		metrics.add(EuclideanRGB.create());
		metrics.add(HumanFriendlyRGB.create());
		metrics.add(EuclideanLab.create());

		PrintWriter calibrationLog = new PrintWriter(
				inputDirectory.getAbsolutePath() + "/calibration_log.txt");
		Map<String, String> calibrationData = new HashMap<>();
		calibrationData.put("header", "1");
		PrintWriter seedLog = new PrintWriter(
				inputDirectory.getAbsolutePath() + "/seed_log.txt");
		Map<String, String> seedData = new HashMap<>();
		seedData.put("header", "1");

		for (File inputFile : inputDirectory.listFiles()) {
			String mimetype = new MimetypesFileTypeMap().getContentType(inputFile);
			String type = mimetype.split("/")[0];
			if (!type.equals("image")) {
				continue;
			}
			System.out.println(inputFile.getName());
			calibrationData.put("file", inputFile.getName());
			seedData.put("file", inputFile.getName());

			Mat image = Imgcodecs.imread(inputFile.getAbsolutePath(),
					Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);
			Mat mask = new Mat(image.rows(), image.cols(), CvType.CV_8UC1, new Scalar(0));

			Quad quad = f.findColorChecker(image);
			Mat extractedColorChecker = quad.getTransformedField(image);
			ColorChecker checker = new ColorChecker(extractedColorChecker);
			Imgcodecs.imwrite(inputDirectory + "/result/" + "extracted_"
					+ inputFile.getName(), checker.drawSamplePoints());
			Double scale = checker.pixelArea(quad);
			calibrationData.put("scale", scale.toString());
			for (ColorMetric cm : metrics) {
				String metricName = cm.getClass().getSimpleName();
				calibrationData.put("source:" + metricName,
						String.valueOf(checker.getCellColors(extractedColorChecker).
								calculateMetric(cm)));
			}
			for (RegressionModel m : models) {
				String name = m.getClass().getSimpleName();
				System.out.println(name);
				calibrationData.put("model", name);
				seedData.put("model", name);

				Mat calibratedChecker = checker.calibrate(extractedColorChecker, m,
						ColorSpace.RGB, ColorSpace.RGB);
				for (ColorMetric cm : metrics) {
					String metricName = cm.getClass().getSimpleName();
					calibrationData.put("calibrated:" + metricName,
							String.valueOf(checker.getCellColors(calibratedChecker).
									calculateMetric(cm)));
				}
				calibratedChecker.release();

				printMap(calibrationLog, calibrationData);
				Mat calibrated = checker.calibrate(image, m,
						ColorSpace.RGB, ColorSpace.RGB);
				f.fillColorChecker(calibrated, quad);
				Imgcodecs.imwrite(inputDirectory + "/result/" + name +
						"_" + inputFile.getName(), calibrated);

				if (!name.equals("IdentityModel")) {
					mask = Main.getMask(calibrated);
				}
				Mat filtered = filterByMask(calibrated, mask);
				calibrated.release();
			
				Imgcodecs.imwrite(inputDirectory + "/result/" + name +
						"_filtered_" + inputFile.getName(), filtered);
				printSeeds(filtered, seedLog, seedData, scale);
				filtered.release();
			}
			image.release();
			mask.release();
			extractedColorChecker.release();
		}
		calibrationLog.close();
		seedLog.close();
	}
}
