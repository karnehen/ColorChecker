package seedcounter.examples;

import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.List;

import javafx.util.Pair;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import seedcounter.ColorChecker;
import seedcounter.FindColorChecker;
import seedcounter.Helper;
import seedcounter.MatchingModel;
import seedcounter.Quad;
import seedcounter.regression.RegressionModel;
import seedcounter.regression.ThirdOrderRGB;

public class CalculateArea {
	private static final List<String> INPUT_FILES = Arrays.asList(
			"IMG_8182.jpg", "IMG_8228.jpg", "IMG_8371.jpg", "IMG_8372.jpg"
	);
	private static final String REFERENCE_FILE = "reference.png";
	private static final MatchingModel MATCHING_MODEL = new MatchingModel(
			FeatureDetector.SIFT, DescriptorExtractor.SIFT,
			DescriptorMatcher.FLANNBASED, 0.7f
	);
	// targets and ranges
	private static final List<Pair<Scalar, Scalar>> POTATO_TYPES = Arrays.asList(
			new Pair<Scalar, Scalar>(new Scalar(4, 97, 108), new Scalar(50, 100, 80)),
			new Pair<Scalar, Scalar>(new Scalar(17, 67, 232), new Scalar(50, 50, 50)),
			new Pair<Scalar, Scalar>(new Scalar(45, 170, 220), new Scalar(30, 30, 30))
		);

	private static void printSeeds(Mat image, Double scale) {
		List<MatOfPoint> contours = Helper.getContours(image);
		Mat seedBuffer = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);

		for (int i = 0; i < contours.size(); ++i) {
			MatOfPoint contour = contours.get(i);
			Double area = scale * Imgproc.contourArea(contour);
			System.out.println("Object: " + i + "; Area: " + area);
			contour.release();
		}

		seedBuffer.release();
	}

	private static Mat getMask(Mat image) {
		Mat mask = Helper.binarizeSeed(image, POTATO_TYPES);
		Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
				new Size(150, 150));
		Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
		kernel.release();
	
		return mask;
	}

	public static void main(String[] args) throws FileNotFoundException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		FindColorChecker f = new FindColorChecker(REFERENCE_FILE, MATCHING_MODEL);

		RegressionModel model = new ThirdOrderRGB();

		for (String inputFile : INPUT_FILES) {
			System.out.println(inputFile);
			Mat image = Highgui.imread(inputFile,
					Highgui.CV_LOAD_IMAGE_ANYCOLOR | Highgui.CV_LOAD_IMAGE_ANYDEPTH);
	
			Quad quad = f.findColorChecker(image);
			Mat extractedColorChecker = quad.getTransformedField(image);
			ColorChecker checker = new ColorChecker(extractedColorChecker);
	
			Mat calibratedChecker = checker.calibrationBgr(extractedColorChecker, model);
			calibratedChecker.release();
	
			Mat calibrated = checker.calibrationBgr(image, model);
			f.fillColorChecker(calibrated, quad);
	
			Mat mask = getMask(calibrated);
			Mat filtered = Helper.filterByMask(calibrated, mask);
			Highgui.imwrite(inputFile.replaceAll("\\..+", "_output.png"), filtered);
			calibrated.release();
	
			Double scale = checker.pixelArea(quad);
			printSeeds(filtered, scale);
	
			filtered.release();
			image.release();
			mask.release();
			extractedColorChecker.release();
		}
	}
}
