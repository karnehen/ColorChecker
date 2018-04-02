package seedcounter.examples;

import javafx.util.Pair;
import org.apache.commons.io.FileUtils;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.xfeatures2d.SIFT;
import seedcounter.*;
import seedcounter.regression.RegressionFactory;
import seedcounter.regression.RegressionFactory.Order;
import seedcounter.regression.RegressionModel;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

class BackgroundDispersion {
	private static final String INPUT_FILES = "src/seedcounter/examples/background_dispersion_input_files";
	private static final String OUTPUT_DIRECTORY = "src/seedcounter/examples/background_dispersion_output/";
	private static final String RESULT_FILE = "src/seedcounter/examples/background_dispersion_results";
	private static final String REFERENCE_FILE = "reference.png";
	// targets and ranges
	private static final List<Pair<Scalar, Scalar>> POTATO_TYPES = Arrays.asList(
			new Pair<>(new Scalar(4, 97, 108), new Scalar(50, 100, 80)),
			new Pair<>(new Scalar(17, 67, 232), new Scalar(50, 50, 50)),
			new Pair<>(new Scalar(45, 170, 220), new Scalar(30, 30, 30))
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

	private static void iterateColorspaces(String inputFile, PrintWriter outputFile, RegressionModel model, Mat image,
									  ColorChecker checker, Mat beforeSamples, Mat[] beforeClusters) {
		File f = new File(OUTPUT_DIRECTORY + model.getClass().getSimpleName() + "/RGB");
		f.mkdir();
		f = new File(f.getAbsolutePath() + "/" + "RGB");
		f.mkdir();

		Mat calibrated = checker.calibrate(image, model, ColorSpace.RGB, ColorSpace.RGB);
		Mat afterSamples = Helper.getClusteringSamples(calibrated);
		Mat[] afterClusters = Helper.clusterize(afterSamples);

		double dispersionChange = Helper.getBackgroundDispersion(afterSamples, beforeClusters[0]) /
				(Helper.getBackgroundDispersion(beforeSamples, beforeClusters[0]) + 1e-5);
		afterSamples.release();

		outputFile.println(inputFile + "\t" + model.getClass().getSimpleName() + "\tRGB\tRGB\t" + dispersionChange);
		File input = new File(inputFile);
		Imgcodecs.imwrite(f.getAbsolutePath() + "/" + input.getName(),
						Helper.getBackgroundSegmentation(calibrated, afterClusters));

		afterClusters[0].release();
		afterClusters[1].release();
		calibrated.release();
	}

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		MatchingModel MATCHING_MODEL = new MatchingModel(
			SIFT.create(), SIFT.create(),
			DescriptorMatcher.FLANNBASED, 0.7f
		);
		FindColorChecker findColorChecker = new FindColorChecker(REFERENCE_FILE, MATCHING_MODEL);

		List<String> inputFiles = null;
		try {
			inputFiles = FileUtils.readLines(new File(INPUT_FILES), "utf-8");
		} catch (IOException e) {
			System.out.println("Can't read from file " + INPUT_FILES);
			System.exit(1);
		}

		PrintWriter outputFile = null;
		try {
			outputFile = new PrintWriter(RESULT_FILE);
		} catch (FileNotFoundException e) {
			System.out.println("Can't write to file " + RESULT_FILE);
			System.exit(1);
		}
		outputFile.println("file\tmodel\tfeatures_space\ttarget_space\tdispersion_change");

		File outputDirectory = new File(OUTPUT_DIRECTORY);
		outputDirectory.mkdir();

		for (String inputFile : inputFiles) {
			Mat image = Imgcodecs.imread(inputFile,
					Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);

			Quad quad = findColorChecker.findColorChecker(image);
			Mat extractedColorChecker = quad.getTransformedField(image);
			ColorChecker checker = new ColorChecker(extractedColorChecker);

		    for (Order order : Order.values()) {
				RegressionModel model = RegressionFactory.createModel(order, false);
				File modelDirectory = new File(OUTPUT_DIRECTORY + model.getClass().getSimpleName());
				modelDirectory.mkdir();

				Mat beforeSamples = Helper.getClusteringSamples(image);
				Mat[] beforeClusters = Helper.clusterize(beforeSamples);

				iterateColorspaces(inputFile, outputFile, model, image, checker, beforeSamples, beforeClusters);

				beforeSamples.release();
				beforeClusters[0].release();
				beforeClusters[1].release();
			}

			image.release();
			extractedColorChecker.release();
		}

		outputFile.close();
	}
}
