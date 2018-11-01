package seedcounter.common;

import javafx.util.Pair;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.PrintWriter;
import java.util.*;
import java.util.stream.Collectors;

public class SeedUtils {
    public SeedUtils() {
        this(0.5, 150.0, true);
    }

    public SeedUtils(double threshold, double whiteThreshold, boolean filterByArea) {
        this.threshold = threshold;
        this.whiteThreshold = whiteThreshold;
        this.filterByArea = filterByArea;
    }

    private final double MIN_AREA = 5.0;
    private final double MAX_AREA = 30.0;
    private final double BRIGHTNESS_PERCENTILE = 10.0;

    private final double threshold;
    private final double whiteThreshold;
    private final boolean filterByArea;

    private int seedNumber = 0;
    private int xOffset = 0;
    private int yOffset = 0;

    public double getThreshold() {
        return threshold;
    }

    public double getWhiteThreshold() {
        return whiteThreshold;
    }

    public boolean getFileterByArea() {
        return filterByArea;
    }

    public void setSeedNumber(int seedNumber) {
        this.seedNumber = seedNumber;
    }

    public int getSeedNumber() {
        return seedNumber;
    }

    public void setXOffset(int xOffset) {
        this.xOffset = xOffset;
    }

    public int getXOffset() {
        return xOffset;
    }

    public void setYOffset(int yOffset) {
        this.yOffset = yOffset;
    }

    public int getYOffset() {
        return yOffset;
    }

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
        return Helper.filterByMask(image, mask);
    }

    public int printSeeds(Mat image, Mat imageForFilter, PrintWriter writer,
                                  Map<String, String> data, Double scale) {
        List<MatOfPoint> contours = Helper.getContours(imageForFilter);
        Mat seedBuffer = Mat.zeros(image.rows(), image.cols(), CvType.CV_8UC1);

        for (MatOfPoint contour : contours) {
            Double area = scale * Imgproc.contourArea(contour);
            if (!filterByArea || (area < MAX_AREA && area > MIN_AREA)) {
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

    private List<Map<String,String>> getSeedData(MatOfPoint contour, Mat image, Mat imageForFilter, Mat seedBuffer) {
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
                        map.put("x", String.valueOf(x + xOffset));
                        map.put("y", String.valueOf(y + yOffset));
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

            if (percentile.evaluate(BRIGHTNESS_PERCENTILE) > whiteThreshold) {
                result.clear();
            }
        }

        return result;
    }

    private void printSeedData(Map<String,String> data, List<Map<String,String>> seedData, PrintWriter writer) {
        for (Map<String,String> map : seedData) {
            for (String key : map.keySet().stream()
                    .sorted().collect(Collectors.toList())) {
                data.put(key, map.get(key));
            }
            printMap(writer, data);
        }
    }

    private void printMap(PrintWriter writer, Map<String, String> map) {
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
