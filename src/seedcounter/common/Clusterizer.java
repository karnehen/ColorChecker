package seedcounter.common;

import org.opencv.core.*;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Clusterizer {
    private final int clusters;

    public Clusterizer(int clusters) {
        this.clusters = clusters;
    }

    /* takes:
     *   source image points reshaped into (rows*cols, channels, 1),
     * 	 cluster labels (rows*cols, 1, 1)
     */
    public double getBackgroundVariance(Mat samples, Mat labels) {
        List<Map<String,Double>> clusterStatistics = new ArrayList<>();
        for (int i = 0; i < clusters; ++i) {
            clusterStatistics.add(Stream.of(
                    new SimpleEntry<>("second_moment", 0.0),
                    new SimpleEntry<>("count", 0.0)
            ).collect(Collectors.toMap(SimpleEntry::getKey, SimpleEntry::getValue)));
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
    public Mat getBackgroundSegmentation(Mat image, Mat[] clusters) {
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
    public Mat getClusteringSamples(Mat image) {
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
    public Mat[] clusterize(Mat samples) {
        return clusterize(samples, 3, 5, 1e-5);
    }

    public Mat[] clusterize(Mat samples, int attempts, int maxIterations, double epsilon) {
        Mat labels = new Mat(samples.rows(), 1, CvType.CV_8U);
        Mat centroids = new Mat(clusters, 1, CvType.CV_32F);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS | TermCriteria.MAX_ITER,
                maxIterations, epsilon);

        Core.kmeans(samples, clusters, labels, criteria, attempts, Core.KMEANS_RANDOM_CENTERS, centroids);

        return new Mat[] {labels, centroids};
    }
}
