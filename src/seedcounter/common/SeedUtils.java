package seedcounter.common;

import javafx.util.Pair;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.PrintWriter;
import java.util.*;

public class SeedUtils {
    // targets and ranges
    public static final List<Pair<Scalar, Scalar>> SEED_TYPES = Arrays.asList(
            new Pair<>(new Scalar(4, 97, 108), new Scalar(50, 100, 80)),
            new Pair<>(new Scalar(17, 67, 232), new Scalar(50, 50, 50))
        );

    public static Mat getMask(Mat image, Double scale) {
        Mat mask = Helper.binarizeSeed(image, SEED_TYPES);
        Mat whiteMask = Helper.whiteThreshold(image);
        Core.bitwise_and(mask, whiteMask, mask);
        whiteMask.release();

        int kernelSize = scale == null ? 10 : (int)(1.5 / Math.sqrt(scale));
        if (kernelSize < 10) {
            kernelSize = 10;
        }
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(kernelSize, kernelSize));

        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
        kernel.release();

        return mask;
    }

    public static Mat filterByMask(Mat image, Mat mask) {
        Mat filtered = Helper.filterByMask(image, mask);
        Range rows = new Range(filtered.rows() / 4, 3 * filtered.rows() / 4);
        Range cols = new Range(filtered.cols() / 4, 3 * filtered.cols() / 4);

        Mat result = new Mat(filtered, rows, cols);
        filtered.release();

        return result;
    }

    public static List<Map<String,String>> getSeedData(MatOfPoint contour, Mat image, Mat imageForFilter,
                                                       Mat seedBuffer) {
        return getSeedData(contour, image, imageForFilter, seedBuffer, 0.5, 150.0);
    }

    public static List<Map<String,String>> getSeedData(MatOfPoint contour, Mat image, Mat imageForFilter, Mat seedBuffer,
                                                       double threshold, double whiteThreshold) {
        Imgproc.drawContours(seedBuffer, Collections.singletonList(contour), 0,
                new Scalar(255.0), Core.FILLED);
        List<Map<String,String>> result = new ArrayList<>();

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

        List<Double> minChannelValues = new ArrayList<>();

        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                if (seedBuffer.get(y, x)[0] > 0.0) {
                    double[] color = image.get(y, x);
                    double[] colorForFilter = imageForFilter.get(y, x);
                    if (color[0] + color[1] + color[2] > 0.0) {
                        Map<String,String> map = new HashMap<>();
                        map.put("x", String.valueOf(x));
                        map.put("y", String.valueOf(y));
                        map.put("blue", String.valueOf(color[0]));
                        map.put("green", String.valueOf(color[1]));
                        map.put("red", String.valueOf(color[2]));
                        minChannelValues.add(Math.min(colorForFilter[0], Math.min(colorForFilter[1], colorForFilter[2])));
                        result.add(map);
                    }
                }
            }
        }

        Imgproc.drawContours(seedBuffer, Collections.singletonList(contour), 0,
                new Scalar(0.0), Core.FILLED);

        if (result.size() / (maxX - minX + 1.0) / (maxY - minY + 1.0) < threshold) {
            result.clear();
        } else {
            Percentile percentile = new Percentile();
            percentile.setData(minChannelValues.stream().mapToDouble(x -> x).toArray());

            if (percentile.evaluate(10.0) > whiteThreshold) {
                result.clear();
            }
        }

        return result;
    }

    public static int printSeeds(Mat image, Mat imageForFilter, PrintWriter writer,
                                  Map<String, String> data, Double scale) {
        return printSeeds(image, imageForFilter, writer, data, scale, 0);
    }

    public static int printSeeds(Mat image, Mat imageForFilter, PrintWriter writer,
                                  Map<String, String> data, Double scale, int seedNumber) {
        List<MatOfPoint> contours = Helper.getContours(imageForFilter);
        Mat seedBuffer = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);

        for (MatOfPoint contour : contours) {
            Double area = scale * Imgproc.contourArea(contour);
            if (area < 30.0 && area > 5.0) {
                List<Map<String,String>> seedData = getSeedData(contour, image, imageForFilter, seedBuffer);
                if (!seedData.isEmpty()) {
                    data.put("seed_number", String.valueOf(seedNumber++));
                    data.put("area", area.toString());
                    printSeedData(data, seedData, writer);
                }
            }
            contour.release();
        }

        seedBuffer.release();

        return seedNumber;
    }

    private static void printSeedData(Map<String,String> data, List<Map<String,String>> seedData, PrintWriter writer) {
        for (Map<String,String> map : seedData) {
            for (String key : map.keySet()) {
                data.put(key, map.get(key));
            }
            printMap(writer, data);
        }
    }

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
}
