package seedcounter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.activation.MimetypesFileTypeMap;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Range;
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

	private static Mat filterSeeds(FindColorChecker f, Mat image) {
		Mat mask = f.binarizeSeed(image);
		Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(50, 50));
		Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
		Mat whiteMask = f.whiteThreshold(image);
		kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(10, 10));
		Imgproc.morphologyEx(whiteMask, whiteMask, Imgproc.MORPH_CLOSE, kernel);

		List<Mat> channels = new ArrayList<Mat>();
		Core.split(image, channels);
		for (int i = 0; i < 3; ++i) {
			Mat c = channels.get(i);
			Core.bitwise_and(c, mask, c);
			Core.bitwise_and(c, whiteMask, c);
			channels.set(i, c);
		}
		Mat filtered = new Mat(image.rows(), image.cols(), CvType.CV_8UC3);
		Core.merge(channels, filtered);
		Range rows = new Range(filtered.rows() / 4, 3 * filtered.rows() / 4);
		Range cols = new Range(filtered.cols() / 4, 3 * filtered.cols() / 4);
		return new Mat(filtered, rows, cols);
	}

	public static void main(String[] args) throws FileNotFoundException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		FindColorChecker f = new FindColorChecker(REFERENCE_FILE, MATCHING_MODELS.get(1));

		File inputDirectory = new File(INPUT_DIRECTORY);
		PrintWriter log = new PrintWriter(inputDirectory.getAbsolutePath() + "/log.txt");
		new File(inputDirectory.getAbsolutePath() + "/result").mkdir();

		List<RegressionModel> models = new ArrayList<RegressionModel>();
		models.add(new SimpleRGB());
		models.add(new SecondOrderRGB());
		models.add(new ThirdOrderRGB());
		List<ColorMetric> metrics = new ArrayList<ColorMetric>();
		metrics.add(EuclideanRGB.create());
		metrics.add(HumanFriendlyRGB.create());
		metrics.add(EuclideanLab.create());

		for (File inputFile : inputDirectory.listFiles()) {
			String mimetype = new MimetypesFileTypeMap().getContentType(inputFile);
			String type = mimetype.split("/")[0];
			if (!type.equals("image")) {
				continue;
			}
			System.out.println(inputFile);
			log.println(inputFile);

			Mat image = Highgui.imread(inputFile.getAbsolutePath(),
					Highgui.CV_LOAD_IMAGE_ANYCOLOR | Highgui.CV_LOAD_IMAGE_ANYDEPTH);

			Quad quad = f.findColorChecker(image);
			Mat extractedColorChecker = quad.getTransformedField(image);
			Highgui.imwrite(inputDirectory + "/result/" + "extracted_" + inputFile.getName(), extractedColorChecker);
			ColorChecker checker = new ColorChecker(extractedColorChecker);
			for (ColorMetric cm : metrics) {
				String metricName = cm.getClass().getName();
				log.println(metricName + ": " +
						checker.getCellColors(extractedColorChecker).calculateMetric(cm));
			}
			for (RegressionModel m : models) {
				String name = m.getClass().getName();
				log.println(name);
				System.out.println(name);
				Mat calibratedChecker = checker.calibrationBgr(extractedColorChecker, m);
				for (ColorMetric cm : metrics) {
					String metricName = cm.getClass().getName();
					log.println(metricName + ": " +
							checker.getCellColors(calibratedChecker).calculateMetric(cm));
				}
				Mat calibrated = checker.calibrationBgr(image, m);
				f.fillColorChecker(calibrated, quad);
				Highgui.imwrite(inputDirectory + "/result/" + name + "_" + inputFile.getName(), calibrated);

				Mat filtered = filterSeeds(f, calibrated);
				Highgui.imwrite(inputDirectory + "/result/" + name + "_filtered_" + inputFile.getName(), filtered);
				PrintWriter seedColors = new PrintWriter(inputDirectory + "/result/" + name + "_colors_" + inputFile.getName() + ".txt");
				for (int r = 0; r < filtered.rows(); ++r) {
					for (int c = 0; c < filtered.cols(); ++c) {
						double[] color = filtered.get(r, c);
						if (color[0] + color[1] + color[2] > 0.0) {
							seedColors.println(color[0] + " " + color[1] + " " + color[2]);
						}
					}
				}
				seedColors.close();
				break;
			}
			break;
		}
		log.close();
	}
}
