package seedcounter;

import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javafx.util.Pair;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class Helper {

	public static Mat whiteThreshold(Mat image) {
		Mat result = image.clone();
		Imgproc.cvtColor(result, result, Imgproc.COLOR_BGR2GRAY);
		Imgproc.threshold(result, result, 200.0, 255.0, Imgproc.THRESH_BINARY_INV);
	
		return result;
	}

	public static Mat binarizeSeed(Mat image,
			List<Pair<Scalar, Scalar>> targetsAndRanges) { 
		HSVBinarization hsv = new HSVBinarization(targetsAndRanges);
		return hsv.apply(image);
	}

	public static Mat filterByMask(Mat image, Mat mask) {
		List<Mat> channels = new ArrayList<>();
		Core.split(image, channels);
		for (int i = 0; i < 3; ++i) {
			Mat c = channels.get(i);
			Core.bitwise_and(c, mask, c);
			channels.set(i, c);
		}
		Mat filtered = new Mat(image.rows(), image.cols(), CvType.CV_8UC3);
		Core.merge(channels, filtered);
		for (Mat c : channels) {
			c.release();
		}

		return filtered;
	}

	public static List<MatOfPoint> getContours(Mat image) {
		Mat gray = new Mat();
		Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY);
		List<MatOfPoint> contours = new ArrayList<>();
	    Mat hierarchy = new Mat();
		Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
	
		gray.release();
		hierarchy.release();
		return contours;
	}

	public static double getBackgroundDispersion(Mat image) {
		Mat[] clusters = clusterize(image);
		Mat samples = clusters[0];
		Mat labels = clusters[1];
		Mat centroids = clusters[2];

		List<Map<String,Double>> clusterStatistics = new ArrayList<>();
		for (int i = 0; i < centroids.rows(); ++i) {
			clusterStatistics.add(Stream.of(
					new SimpleEntry<>("rss", 0.0),
					new SimpleEntry<>("count", 0.0)
			).collect(Collectors.toMap((e) -> e.getKey(), (e) -> e.getValue())));
		}

		for (int row = 0; row < labels.rows(); ++row) {
			int label = (int)labels.get(row, 0)[0];
			Map<String,Double> map = clusterStatistics.get(label);

			for (int col = 0; col < samples.cols(); ++col) {
				map.put("rss", map.get("rss") +
					Math.pow(centroids.get(label, col)[0] - samples.get(row, col)[0], 2));
			}
			map.put("count", map.get("count") + 1.0);
		}

		labels.release();
		centroids.release();
		samples.release();

		Map<String,Double> maxCluster = clusterStatistics.get(0);
		for (Map<String,Double> c : clusterStatistics) {
			if (c.get("count") > maxCluster.get("count")) {
				maxCluster = c;
			}
		}

		return maxCluster.get("rss") / maxCluster.get("count");
	}

	public static Mat getBackgroundSegmentation(Mat image) {
		Mat[] clusters = clusterize(image);
		Mat samples = clusters[0];
		Mat labels = clusters[1];
		Mat centroids = clusters[2];

		for (int row = 0; row < labels.rows(); ++row) {
			int label = (int)labels.get(row, 0)[0];
			Mat samplesRow = samples.row(row);
			centroids.row(label).copyTo(samplesRow);
		}
		labels.release();
		centroids.release();

		samples = samples.reshape(image.channels(), image.rows());
		samples.convertTo(samples, image.type());

		return samples;
	}

	/* returns: {
	 * 	 source image points reshaped into (rows*cols, channels, 1),
	 * 	 cluster labels (rows*cols, 1, 1),
	 * 	 cluster centroids (CLUSTERS, channels, 1)
	 */
	private static Mat[] clusterize(Mat image) {
		final int CLUSTERS = 3;
		final int ATTEMPTS = 3;

		Mat samples = image.reshape(1, image.rows() * image.cols());
		samples.convertTo(samples, CvType.CV_32F);

		Mat labels = new Mat(samples.rows(), 1, CvType.CV_8U);
		Mat centroids = new Mat(CLUSTERS, 1, CvType.CV_32F);
		TermCriteria criteria = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER,
				5, 1e-5);

		Core.kmeans(samples, CLUSTERS, labels, criteria, ATTEMPTS, Core.KMEANS_RANDOM_CENTERS, centroids);

		return new Mat[] {samples, labels, centroids};
	}
}
