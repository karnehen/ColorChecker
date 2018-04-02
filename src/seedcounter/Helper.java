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
	private static final int CLUSTERS = 3;

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

	/* takes:
	 *   source image points reshaped into (rows*cols, channels, 1),
	 * 	 cluster labels (rows*cols, 1, 1)
	 */
	public static double getBackgroundDispersion(Mat samples, Mat labels) {
		List<Map<String,Double>> clusterStatistics = new ArrayList<>();
		for (int i = 0; i < CLUSTERS; ++i) {
			clusterStatistics.add(Stream.of(
					new SimpleEntry<>("second_moment", 0.0),
					new SimpleEntry<>("count", 0.0)
			).collect(Collectors.toMap((e) -> e.getKey(), (e) -> e.getValue())));
			for (int col = 0; col < samples.cols(); ++col) {
				clusterStatistics.get(i).put("sum_channel_" + col, 0.0);
			}
		}

		for (int row = 0; row < labels.rows(); ++row) {
			int label = (int)labels.get(row, 0)[0];
			Map<String,Double> map = clusterStatistics.get(label);

			for (int col = 0; col < samples.cols(); ++col) {
				map.put("second_moment", map.get("second_moment") + Math.pow(samples.get(row, col)[0], 2));
				map.put("sum_channel_" + col, map.get("sum_channel_" + col) + samples.get(row, col)[0]);
			}
			map.put("count", map.get("count") + 1.0);
		}

		Map<String,Double> maxCluster = clusterStatistics.get(0);
		for (Map<String,Double> c : clusterStatistics) {
			if (c.get("count") > maxCluster.get("count")) {
				maxCluster = c;
			}
		}

		double result = maxCluster.get("second_moment") / maxCluster.get("count");

		for (int col = 0; col < samples.cols(); ++col) {
			result -= Math.pow(maxCluster.get("sum_channel_" + col) / maxCluster.get("count"), 2);
		}

		return result;
	}

	/* takes:
	 *   source image,
	 * 	 array {
	 * 	 	cluster labels (rows*cols, 1, 1),
	 * 	 	cluster centroids (CLUSTERS, channels, 1)
	 * 	 }
	 */
	public static Mat getBackgroundSegmentation(Mat image, Mat[] clusters) {
		Mat samples = getClusteringSamples(image);
		Mat labels = clusters[0];
		Mat centroids = clusters[1];

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

	// returns: source image points reshaped into (rows*cols, channels, 1)
	public static Mat getClusteringSamples(Mat image) {
		Mat samples = image.reshape(1, image.rows() * image.cols());
		samples.convertTo(samples, CvType.CV_32F);

		return samples;
	}

	/* takes: source image points reshaped into (rows*cols, channels, 1)
	 * returns: {
	 * 	 cluster labels (rows*cols, 1, 1),
	 * 	 cluster centroids (CLUSTERS, channels, 1)
	 * }
	 */
	public static Mat[] clusterize(Mat samples) {
		final int ATTEMPTS = 3;

		Mat labels = new Mat(samples.rows(), 1, CvType.CV_8U);
		Mat centroids = new Mat(CLUSTERS, 1, CvType.CV_32F);
		TermCriteria criteria = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER,
				5, 1e-5);

		Core.kmeans(samples, CLUSTERS, labels, criteria, ATTEMPTS, Core.KMEANS_RANDOM_CENTERS, centroids);

		return new Mat[] {labels, centroids};
	}
}
