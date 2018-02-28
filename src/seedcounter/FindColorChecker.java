package seedcounter;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;


import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


public class FindColorChecker {
	private final Mat referenceImage;
	private final MatchingModel matchingModel;
	private final MatOfKeyPoint referenceKeypoints;
	private final MatOfKeyPoint referenceDescriptors;
	private final FeatureDetector detector;
	private final DescriptorExtractor extractor;

	public FindColorChecker(String referenceFile, MatchingModel matchingModel) {
		referenceImage = Imgcodecs.imread(referenceFile,
				Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);
		this.matchingModel = matchingModel;

		referenceKeypoints = new MatOfKeyPoint();
		detector = FeatureDetector.create(this.matchingModel.getDetector());
		detector.detect(referenceImage, referenceKeypoints);

		referenceDescriptors = new MatOfKeyPoint();
		extractor = DescriptorExtractor.create(this.matchingModel.getExtractor());
		extractor.compute(referenceImage, referenceKeypoints, referenceDescriptors);
	}

	public Quad findColorChecker(Mat image) {
		MatOfKeyPoint keypoints = new MatOfKeyPoint();
		detector.detect(image, keypoints);

		MatOfKeyPoint descriptors = new MatOfKeyPoint();
		extractor.compute(image, keypoints, descriptors);

		LinkedList<DMatch> goodMatches = getGoodMatches(descriptors);

		Mat homography = getHomography(keypoints, goodMatches);

		return getQuad(homography);
	}

	private LinkedList<DMatch> getGoodMatches(MatOfKeyPoint descriptors) {
		List<MatOfDMatch> matches = new LinkedList<>();
		DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(
				matchingModel.getMatcher());
		descriptorMatcher.knnMatch(referenceDescriptors, descriptors, matches, 2);

		LinkedList<DMatch> goodMatches = new LinkedList<>();

		for (MatOfDMatch matofDMatch : matches) {
			DMatch[] dmatcharray = matofDMatch.toArray();
			DMatch m1 = dmatcharray[0];
			DMatch m2 = dmatcharray[1];

			if (m1.distance <= m2.distance * matchingModel.getThreshold()) {
				goodMatches.addLast(m1);
			}
		}

		return goodMatches;
	}

	private Mat getHomography(MatOfKeyPoint keypoints, LinkedList<DMatch> goodMatches) {
		List<KeyPoint> referenceKeypointlist = referenceKeypoints.toList();
		List<KeyPoint> keypointlist = keypoints.toList();

		LinkedList<Point> referencePoints = new LinkedList<>();
		LinkedList<Point> points = new LinkedList<>();

		for (DMatch goodMatch : goodMatches) {
			referencePoints.addLast(referenceKeypointlist.get(goodMatch.queryIdx).pt);
			points.addLast(keypointlist.get(goodMatch.trainIdx).pt);
		}

		MatOfPoint2f referenceMatOfPoint2f = new MatOfPoint2f();
		referenceMatOfPoint2f.fromList(referencePoints);
		MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
		matOfPoint2f.fromList(points);

		return Calib3d.findHomography(referenceMatOfPoint2f, matOfPoint2f, Calib3d.RANSAC, 3);
	}

	private Quad getQuad(Mat homography) {
		Mat referenceCorners = new Mat(4, 1, CvType.CV_32FC2);
		Mat corners = new Mat(4, 1, CvType.CV_32FC2);

		referenceCorners.put(0, 0, -0.01 * referenceImage.cols(), -0.01 * referenceImage.rows());
		referenceCorners.put(1, 0, 1.01 * referenceImage.cols(), -0.01 * referenceImage.rows());
		referenceCorners.put(2, 0, 1.01 * referenceImage.cols(), 1.01 * referenceImage.rows());
		referenceCorners.put(3, 0, -0.01 * referenceImage.cols(), 1.01 * referenceImage.rows());

		Core.perspectiveTransform(referenceCorners, corners, homography);

		return new Quad(new Point(corners.get(0, 0)),new Point(corners.get(1, 0)),
				new Point(corners.get(2, 0)), new Point(corners.get(3, 0)));
	}

	public void fillColorChecker(Mat image, Quad quad) {
		MatOfPoint points = new MatOfPoint();

		points.fromArray(quad.getPoints());
		Imgproc.fillConvexPoly(image, points, getBackgroundColor(image, quad));
	}

	private Scalar getBackgroundColor(Mat image, Quad quad) {
		List<double[]> colors = new ArrayList<>();
		double sumBlue = 0.0;
		double sumGreen = 0.0;
		double sumRed = 0.0;

		while (colors.size() < 1000) {
			double x = ThreadLocalRandom.current().nextDouble(0.0, image.cols());
			double y = ThreadLocalRandom.current().nextDouble(0.0, image.rows());

			if (image.get((int) x, (int) y) != null && !quad.isInside(new Point(x, y))) {
				colors.add(image.get((int) x, (int) y));
				sumBlue += colors.get(colors.size() - 1)[0];
				sumGreen += colors.get(colors.size() - 1)[1];
				sumRed += colors.get(colors.size() - 1)[2];
			}
		}
		double meanBlue = sumBlue / colors.size();
		double meanGreen = sumGreen / colors.size();
		double meanRed = sumRed / colors.size();

		for (int i = 0; i < 10; ++i) {
			List<double[]> buffer = new ArrayList<>();
			sumBlue = 0.0;
			sumGreen = 0.0;
			sumRed = 0.0;
			for (double[] c : colors) {
				if (Math.abs(c[0] - meanBlue) <= 25 &&
					Math.abs(c[1] - meanGreen) <= 25 &&
					Math.abs(c[2] - meanRed) <= 25) {
					sumBlue += c[0];
					sumGreen += c[1];
					sumRed += c[2];
					buffer.add(c);
				}
			}
			colors = buffer;
			meanBlue = sumBlue / colors.size();
			meanGreen = sumGreen / colors.size();
			meanRed = sumRed / colors.size();
		}

		return new Scalar(meanBlue, meanGreen, meanRed);
	}
}
