package seedcounter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.activation.MimetypesFileTypeMap;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import seedcounter.colormetric.ColorMetric;
import seedcounter.colormetric.EuclideanLab;
import seedcounter.colormetric.EuclideanRGB;
import seedcounter.colormetric.HumanFriendlyRGB;
import seedcounter.regression.IdentityModel;
import seedcounter.regression.RegressionModel;
import seedcounter.regression.SecondOrderRGB;
import seedcounter.regression.SimpleRGB;
import seedcounter.regression.ThirdOrderRGB;

public class Main {
	private static final String INPUT_DIRECTORY = "../../photos/ASUS_Z00ED";
	private static final String REFERENCE_FILE = "reference.png";
	private static final List<MatchingModel> MATCHING_MODELS = Arrays.asList(
			new MatchingModel(FeatureDetector.SURF, DescriptorExtractor.SURF,
					DescriptorMatcher.FLANNBASED, 0.7f),
			new MatchingModel(FeatureDetector.SIFT, DescriptorExtractor.SIFT,
					DescriptorMatcher.FLANNBASED, 0.7f),
			new MatchingModel(FeatureDetector.ORB, DescriptorExtractor.ORB,
					DescriptorMatcher.BRUTEFORCE_HAMMING, 0.9f)
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

	private static Mat getMask(FindColorChecker f, Mat image) {
		Mat mask = f.binarizeSeed(image);
		Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(50, 50));
		Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
		Mat whiteMask = f.whiteThreshold(image);
		kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(10, 10));
		Imgproc.morphologyEx(whiteMask, whiteMask, Imgproc.MORPH_CLOSE, kernel);
		Core.bitwise_and(mask, whiteMask, mask);

		return mask;
	}

	private static Mat filterByMask(Mat image, Mat mask) {
		List<Mat> channels = new ArrayList<Mat>();
		Core.split(image, channels);
		for (int i = 0; i < 3; ++i) {
			Mat c = channels.get(i);
			Core.bitwise_and(c, mask, c);
			channels.set(i, c);
		}
		Mat filtered = new Mat(image.rows(), image.cols(), CvType.CV_8UC3);
		Core.merge(channels, filtered);
		Range rows = new Range(filtered.rows() / 4, 3 * filtered.rows() / 4);
		Range cols = new Range(filtered.cols() / 4, 3 * filtered.cols() / 4);
		return new Mat(filtered, rows, cols);
	}

	private static List<MatOfPoint> getContours(Mat image) {
		Mat gray = new Mat();
		Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY);
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
		Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

		System.out.println(contours.size());
		return contours;
	}

	private static void printSingleSeed(MatOfPoint contour, Mat image, Mat seedBuffer,
			PrintWriter writer, Map<String, String> data) {
		data.put("area", String.valueOf(Imgproc.contourArea(contour)));
		Imgproc.drawContours(seedBuffer, Arrays.asList(contour), 0,
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
		Imgproc.drawContours(seedBuffer, Arrays.asList(contour), 0, new Scalar(0.0));
	}

	private static void printSeeds(Mat image, PrintWriter writer,
			Map<String, String> data) {
		List<MatOfPoint> contours = getContours(image);
		Mat seedBuffer = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);

		for (int i = 0; i < contours.size(); ++i) {
			data.put("seed_number", String.valueOf(i));
			printSingleSeed(contours.get(i), image, seedBuffer, writer, data);
		}
	}

	public static void main(String[] args) throws FileNotFoundException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		FindColorChecker f = new FindColorChecker(REFERENCE_FILE, MATCHING_MODELS.get(1));

		File inputDirectory = new File(INPUT_DIRECTORY);
		new File(inputDirectory.getAbsolutePath() + "/result").mkdir();

		List<RegressionModel> models = new ArrayList<RegressionModel>();
		models.add(new SimpleRGB());
		models.add(new SecondOrderRGB());
		models.add(new ThirdOrderRGB());
		models.add(new IdentityModel());
		List<ColorMetric> metrics = new ArrayList<ColorMetric>();
		metrics.add(EuclideanRGB.create());
		metrics.add(HumanFriendlyRGB.create());
		metrics.add(EuclideanLab.create());

		PrintWriter calibrationLog = new PrintWriter(
				inputDirectory.getAbsolutePath() + "/calibration_log.txt");
		Map<String, String> calibrationData = new HashMap<String, String>();
		calibrationData.put("header", "1");
		PrintWriter seedLog = new PrintWriter(
				inputDirectory.getAbsolutePath() + "/seed_log.txt");
		Map<String, String> seedData = new HashMap<String, String>();
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

			Mat image = Highgui.imread(inputFile.getAbsolutePath(),
					Highgui.CV_LOAD_IMAGE_ANYCOLOR | Highgui.CV_LOAD_IMAGE_ANYDEPTH);
			Mat mask = new Mat(image.rows(), image.cols(), CvType.CV_8UC1, new Scalar(0));

			Quad quad = f.findColorChecker(image);
			Mat extractedColorChecker = quad.getTransformedField(image);
			Highgui.imwrite(inputDirectory + "/result/" + "extracted_" + inputFile.getName(), extractedColorChecker);
			ColorChecker checker = new ColorChecker(extractedColorChecker);
			calibrationData.put("scale", checker.pixelArea(quad).toString());
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
				Mat calibratedChecker = checker.calibrationBgr(extractedColorChecker, m);
				for (ColorMetric cm : metrics) {
					String metricName = cm.getClass().getSimpleName();
					calibrationData.put("calibrated:" + metricName,
							String.valueOf(checker.getCellColors(calibratedChecker).
									calculateMetric(cm)));
				}
				printMap(calibrationLog, calibrationData);
				Mat calibrated = checker.calibrationBgr(image, m);
				f.fillColorChecker(calibrated, quad);
				Highgui.imwrite(inputDirectory + "/result/" + name +
						"_" + inputFile.getName(), calibrated);

				if (!name.equals("IdentityModel")) {
					mask = getMask(f, calibrated);
				}
				Mat filtered = filterByMask(calibrated, mask);
				Highgui.imwrite(inputDirectory + "/result/" + name +
						"_filtered_" + inputFile.getName(), filtered);
				printSeeds(filtered, seedLog, seedData);
			}
		}
		calibrationLog.close();
		seedLog.close();
	}
}
