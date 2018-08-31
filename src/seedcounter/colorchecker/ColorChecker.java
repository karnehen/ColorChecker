package seedcounter.colorchecker;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import seedcounter.colormetric.CellColors;
import seedcounter.colormetric.EuclideanLab;
import seedcounter.colormetric.EuclideanRGB;
import seedcounter.common.Quad;
import seedcounter.colormetric.Color;
import seedcounter.regression.ColorSpace;
import seedcounter.regression.RegressionModel;

public class ColorChecker {
    private final static List<List<Scalar>> BGR_REFERENCE_COLORS = Arrays.asList(
            Arrays.asList(new Scalar(171, 191, 99), new Scalar(41, 161, 229), new Scalar(166, 136, 0), new Scalar(50, 50, 50)),
            Arrays.asList(new Scalar(176, 129, 130), new Scalar(62, 189, 160), new Scalar(150, 84, 188), new Scalar(85, 84, 83)),
            Arrays.asList(new Scalar(65, 108, 90), new Scalar(105, 59, 91), new Scalar(22, 200, 238), new Scalar(121, 121, 120)),
            Arrays.asList(new Scalar(157, 123, 93), new Scalar(98, 84, 195), new Scalar(56, 48, 176), new Scalar(161, 161, 160)),
            Arrays.asList(new Scalar(129, 149, 196), new Scalar(168, 92, 72), new Scalar(72, 149, 71), new Scalar(201, 201, 200)),
            Arrays.asList(new Scalar(67, 81, 115), new Scalar(45, 123, 220), new Scalar(147, 62, 43), new Scalar(240, 245, 245))
    );

    private final Mat checkerImage;
    private final List<List<Point>> centers;
    private final Integer xScale;
    private final Integer xColorPatchSize;
    private final Integer yScale;
    private final Integer yColorPatchSize;

    private static final Double REAL_WIDTH = 64.0; // millimeters
    private static final Double REAL_HEIGHT = 108.0; // millimeters
    private static final List<Integer> TOP_INDEXES = Arrays.asList(0, 1, 2);
    private static  final List<Integer> BOTTOM_INDEXES = Arrays.asList(6, 7, 8);
    private static final List<Integer> LEFT_INDEXES = Arrays.asList(0, 3, 6);
    private static final List<Integer> RIGHT_INDEXES = Arrays.asList(2, 5, 8);

    public ColorChecker(Mat image) {
        this(image, true, true);
    }

    /*
     * withCorrectionByReference - try to correct recognized points on ColorChecker palette
     *     by comparing the colors with the reference
     * withCorrectionByDeviation - try to correct recognized points by shifting in opposite direction
     *     from the points having the most deviation from the center
     */
    public ColorChecker(Mat image, boolean withCorrectionByReference, boolean withCorrectionByDeviation) {
        checkerImage = image;
        Integer width = image.width();
        Integer height = image.height();

        xScale = (int) (0.04 * width);
        xColorPatchSize = xScale / 8;
        yScale = (int) (0.02 * height);
        yColorPatchSize = yScale / 8;

        List<Double> xCenters = Arrays.asList(0.143, 0.381, 0.613, 0.862);
        List<Double> yCenters = Arrays.asList(0.160, 0.305, 0.440, 0.580, 0.717, 0.856);

        centers = new ArrayList<>();
        for (int row = 0; row < rowCount(); ++row) {
            List<Point> points = new ArrayList<>();
            for (int col = 0; col < colCount(); ++col) {
                Point point = new Point(xCenters.get(col) * width, yCenters.get(row) * height);
                if (withCorrectionByReference) {
                    point = correctByReference(point, row, col);
                }
                if (withCorrectionByDeviation) {
                    point = correctByDeviation(point);
                }
                points.add(point);
            }
            centers.add(points);
        }
    }

    public Double pixelArea(Quad quad) {
        return REAL_WIDTH * REAL_HEIGHT / quad.getArea();
    }

    public Mat calibrate(Mat srcImage, RegressionModel model,
                         ColorSpace featuresSpace, ColorSpace targetSpace) throws IllegalStateException {
        Mat result = srcImage.clone();
        result.convertTo(result, CvType.CV_64FC3);

        int channels = result.channels();
        int size = (int) result.total() * channels;
        double[] temp = new double[size];
        result.get(0, 0, temp);

        for (int i = 0; i + channels < size; i += channels) {
            featuresSpace.convertFromBGR(DoubleBuffer.wrap(temp, i, channels), true);
        }

        List<DoubleBuffer> train = new ArrayList<>();
        List<DoubleBuffer> answers = new ArrayList<>();

        calculateTrainAndAnswers(featuresSpace, targetSpace, train, answers);

        try {
            model.train(train, answers);
        } catch (SingularMatrixException e) {
            result.release();
            throw new IllegalStateException("Couldn't calibrate colors given this reference");
        }

        for (int i = 0; i + channels < size; i += channels) {
            DoubleBuffer srcColor = DoubleBuffer.wrap(temp, i, channels);
            model.calibrate(srcColor);
        }

        for (int i = 0; i + channels < size; i += channels) {
            targetSpace.convertToBGR(DoubleBuffer.wrap(temp, i, channels));
        }

        result.put(0, 0, temp);
        result.convertTo(result, srcImage.type());

        return result;
    }

    // a wrapper for the getTransformationDeviation method in AbstractOLS class
    public double getTransformationDeviation(RegressionModel model, ColorSpace featuresSpace) throws IllegalStateException {
        List<DoubleBuffer> train = new ArrayList<>();
        List<DoubleBuffer> answers = new ArrayList<>();

        calculateTrainAndAnswers(featuresSpace, featuresSpace, train, answers);

        double deviation;

        try {
            deviation = model.getTransformationDeviance(train, answers);
        } catch (SingularMatrixException e) {
            throw new IllegalStateException("Couldn't calculate the transformation matrix given this reference");
        }

        return deviation;
    }

    private void calculateTrainAndAnswers(ColorSpace featuresSpace, ColorSpace targetSpace,
                                          List<DoubleBuffer> train, List<DoubleBuffer> answers) {
        for (Integer row = 0; row < rowCount(); ++row) {
            for (Integer col = 0; col < colCount(); ++col) {
                List<DoubleBuffer> samplePoints = getSamplePoints(row, col);
                for (DoubleBuffer s : samplePoints) {
                    train.add(featuresSpace.convertFromBGR(s, false));
                    DoubleBuffer referenceColor = DoubleBuffer.wrap(BGR_REFERENCE_COLORS.get(row).get(col).val);
                    answers.add(targetSpace.convertFromBGR(referenceColor, false));
                }
            }
        }
    }

    public double labDeviationFromReference() {
        return getCellColors(checkerImage, false).calculateMetric(new EuclideanLab());
    }

    public CellColors getCellColors(Mat checkerImage) {
        return getCellColors(checkerImage, true);
    }

    public CellColors getCellColors(Mat checkerImage, boolean allPoints) {
        CellColors cellColors = new CellColors();

        for (Integer row = 0; row < rowCount(); ++row) {
            for (Integer col = 0; col < colCount(); ++col) {
                List<DoubleBuffer> actualColors = getSamplePoints(checkerImage, row, col, allPoints);
                DoubleBuffer referenceColor = DoubleBuffer.wrap(BGR_REFERENCE_COLORS.get(row).get(col).val);
                for (DoubleBuffer color : actualColors) {
                    cellColors.addColor(new Color(color), new Color(referenceColor));
                }
            }
        }

        return cellColors;
    }

    private List<DoubleBuffer> getSamplePoints(Integer row, Integer col) {
        return getSamplePoints(checkerImage, row, col, false);
    }

    private List<DoubleBuffer> getSamplePoints(Mat checkerImage, Integer row, Integer col, boolean allPoints) {
        final int STEP = 10;
        final int CHANNELS = 3;

        Point center = centers.get(row).get(col);
        List<Point> surroundingPoints = getSurroundingPoints(center);

        List<DoubleBuffer> points = new ArrayList<>();
        double[] result;

        if (allPoints) {
            int minX = (int) surroundingPoints.get(0).x;
            int minY = (int) surroundingPoints.get(0).y;
            int maxX = (int) surroundingPoints.get(8).x;
            int maxY = (int) surroundingPoints.get(8).y;
            int xSize = (maxX - minX) / STEP + 1;
            int ySize = (maxY - minY) / STEP + 1;
            result = new double[xSize * ySize * CHANNELS];
            int index = 0;

            for (int y = minY; y <= maxY; y += STEP) {
                for (int x = minX; x <= maxX; x += STEP) {
                    double[] color = checkerImage.get(y, x);
                    System.arraycopy(color, 0, result, index, CHANNELS);
                    points.add(DoubleBuffer.wrap(result, index, CHANNELS));
                    index += CHANNELS;
                }
            }
        } else {
            result = new double[surroundingPoints.size() * CHANNELS];
            for (int i = 0; i < surroundingPoints.size(); ++i) {
                Point p = surroundingPoints.get(i);
                double[] color = checkerImage.get((int) p.y, (int) p.x);
                System.arraycopy(color, 0, result, i * 3, CHANNELS);
                points.add(DoubleBuffer.wrap(result, i*CHANNELS, CHANNELS));
            }
        }

        return points;
    }

    public Mat drawSamplePoints() {
        Mat result = checkerImage.clone();
        Scalar red = new Scalar(0, 0, 255);
        Scalar blue = new Scalar(255, 0, 0);

        for (Integer row = 0; row < rowCount(); ++row) {
            for (Integer col = 0; col < colCount(); ++col) {
                Point center = centers.get(row).get(col);
                List<Point> points = getSurroundingPoints(center);
                int i = 0;
                for (Point p : points) {
                    if (i % 2 == 0) {
                        Imgproc.circle(result, p, 10, red, Core.FILLED);
                    } else {
                        Imgproc.circle(result, p, 10, blue, Core.FILLED);
                    }
                    i += 1;
                }
            }
        }

        return result;
    }

    private Point correctByReference(Point center, int row, int col) {
        final int ITERATIONS = 3;
        final double STEP_CHANGE = 1.2;
        final double THRESHOLD = (row <= 1 && col == 3 ? 2.0 : 1.1);
        final double VARIANCE_THRESHOLD = 100.0;
        final double DISTANCE_COEFFICIENT = 1.5;

        return correctByReference(center, row, col, ITERATIONS, STEP_CHANGE, THRESHOLD,
                VARIANCE_THRESHOLD, DISTANCE_COEFFICIENT);
    }

    private Point correctByReference(Point center, int row, int col, int iterations, double stepChange,
                                     double threshold, double varianceThreshold, double distanceCoefficient) {
        final double INFINITY = 1e9;

        Color referenceColor = new Color(DoubleBuffer.wrap(BGR_REFERENCE_COLORS.get(row).get(col).val));
        EuclideanLab metric = new EuclideanLab();
        double xStep = xScale;
        double yStep = yScale;

        for (int iteration = 0; iteration < iterations; ++iteration) {
            List<Point> points = getSurroundingPoints(center);
            int nearestPoint = -1;
            double nearestDistance = INFINITY;

            for (int i = 0; i < points.size(); ++i) {
                Point point = points.get(i);
                int x = (int) (center.x + (point.x - center.x) * distanceCoefficient);
                int y = (int) (center.y + (point.y - center.y) * distanceCoefficient);
                if (getValueVariance(x, y) < varianceThreshold) {
                    Color color = getMeanColor(x, y);
                    double distance = metric.calculate(color, referenceColor);
                    if (distance < nearestDistance) {
                        nearestPoint = i;
                        nearestDistance = distance;
                    }
                }
            }

            if (nearestPoint == -1) {
                return center;
            }
            Point point = points.get(8 - nearestPoint);
            int x = (int) (center.x + (point.x - center.x));
            int y = (int) (center.y + (point.y - center.y));
            Color color = getMeanColor(x, y);
            double oppositeDistance = metric.calculate(color, referenceColor);

            if (oppositeDistance < nearestDistance * threshold) {
                return center;
            }

            if (RIGHT_INDEXES.contains(nearestPoint)) {
                if (checkCorrectness(points, xStep, 0.0)) {
                    center.x += xStep;
                }
                xStep /= stepChange;
            } else if (LEFT_INDEXES.contains(nearestPoint)) {
                if (checkCorrectness(points, -xStep, 0.0)) {
                    center.x -= xStep;
                }
                xStep /= stepChange;
            }

            if (BOTTOM_INDEXES.contains(nearestPoint)) {
                if (checkCorrectness(points, 0.0, yStep)) {
                    center.y += yStep;
                }
                yStep /= stepChange;
            } else if (TOP_INDEXES.contains(nearestPoint)) {
                if (checkCorrectness(points, 0.0, -yStep)) {
                    center.y -= yStep;
                }
                yStep /= stepChange;
            }
        }

        return center;
    }

    private Color getMeanColor(int x, int y) {
        double[] result = {0.0, 0.0, 0.0};

        for (int row = -1; row < 2; ++row) {
            for (int col = -1; col < 2; ++col) {
                double[] color = getColor(x + row * xColorPatchSize, y + col * yColorPatchSize);
                for (int i = 0; i < 3; ++i) {
                    result[i] += color[i];
                }
            }
        }

        for (int i = 0; i < 3; ++i) {
            result[i] /= 9;
        }

        return new Color(DoubleBuffer.wrap(result));
    }

    private double getValueVariance(int x, int y) {
        double firstMoment = 0.0;
        double secondMoment = 0.0;

        for (int row = -1; row < 2; ++row) {
            for (int col = -1; col < 2; ++col) {
                double[] color = getColor(x + row * xColorPatchSize, y + col * yColorPatchSize);
                double value = Math.max(color[0], Math.max(color[1], color[2]));
                firstMoment += value;
                secondMoment += Math.pow(value, 2.0);
            }
        }

        firstMoment /= 9.0;
        secondMoment /= 9.0;

        return secondMoment - Math.pow(firstMoment, 2.0);
    }

    private double[] getColor(int x, int y) {
        if (x < 0) {
            x = 0;
        } else if (x >= checkerImage.cols()) {
            x = checkerImage.cols() - 1;
        }
        if (y < 0) {
            y = 0;
        } else if (y >= checkerImage.rows()) {
            y = checkerImage.rows() - 1;
        }

        return checkerImage.get(y, x);
    }

    private Point correctByDeviation(Point center) {
        final int ITERATIONS = 10;
        final double THRESHOLD = 1.5;
        final double STEP_CHANGE = 1.5;

        return correctByDeviation(center, ITERATIONS, THRESHOLD, STEP_CHANGE);
    }

    private Point correctByDeviation(Point center, int iterations, double threshold, double stepChange) {
        double xStep = xScale;
        double yStep = yScale;

        for (int iteration = 0; iteration < iterations; ++iteration) {
            List<Point> points = getSurroundingPoints(center);
            double top = deviationSum(points, center, TOP_INDEXES);
            double bottom = deviationSum(points, center, BOTTOM_INDEXES);
            double left = deviationSum(points, center, LEFT_INDEXES);
            double right = deviationSum(points, center, RIGHT_INDEXES);
            Point newCenter = center.clone();

            if (left >= right * threshold) {
                if (checkCorrectness(points, xStep, 0.0)) {
                    double newLeft = deviationSum(points, center, LEFT_INDEXES, xStep, 0.0);
                    double newRight = deviationSum(points, center, RIGHT_INDEXES, xStep, 0.0);
                    if (newLeft + newRight < left + right) {
                        newCenter.x += xStep;
                    }
                }
                xStep /= stepChange;
            } else if (right >= left * threshold) {
                if (checkCorrectness(points, -xStep, 0.0)) {
                    double newLeft = deviationSum(points, center, LEFT_INDEXES, -xStep, 0.0);
                    double newRight = deviationSum(points, center, RIGHT_INDEXES, -xStep, 0.0);
                    if (newLeft + newRight < left + right) {
                        newCenter.x -= xStep;
                    }
                }
                xStep /= stepChange;
            }

            if (top >= bottom * threshold) {
                if (checkCorrectness(points, 0.0, yStep)) {
                    double newTop = deviationSum(points, center, TOP_INDEXES, 0.0, yStep);
                    double newBottom = deviationSum(points, center, BOTTOM_INDEXES, 0.0, yStep);
                    if (newTop + newBottom < top + bottom) {
                        newCenter.y += yStep;
                    }
                }
                yStep /= stepChange;
            } else if (bottom >= top * threshold) {
                if (checkCorrectness(points, 0.0, -yStep)) {
                    double newTop = deviationSum(points, center, TOP_INDEXES, 0.0, -yStep);
                    double newBottom = deviationSum(points, center, BOTTOM_INDEXES, 0.0, -yStep);
                    if (newTop + newBottom < top + bottom) {
                        newCenter.y -= yStep;
                    }
                }
                yStep /= stepChange;
            }

            center = newCenter;
        }

        return center;
    }

    private double deviationSum(List<Point> points, Point center, List<Integer> indexes) {
        return deviationSum(points, center, indexes, 0.0, 0.0);
    }

    private double deviationSum(List<Point> points, Point center, List<Integer> indexes, double xStep, double yStep) {
        Color centerColor = new Color(DoubleBuffer.wrap(checkerImage.get((int) (center.y + yStep),
                (int) (center.x + xStep))));
        EuclideanRGB metric = new EuclideanRGB();

        return indexes.stream().map(x -> {
            Point point = points.get(x);
            Color color = new Color(DoubleBuffer.wrap(
                    checkerImage.get((int) (point.y + yStep), (int) (point.x + xStep))));
            return metric.calculate(centerColor, color);
        }).reduce(0.0, (x,y) -> x + y);
    }

    private boolean checkCorrectness(List<Point> points, double xStep, double yStep) {
        for (Point p : points) {
            if (p.x + xStep < 0 || p.x + xStep >= checkerImage.cols()
                    || p.y + yStep < 0 || p.y + yStep >= checkerImage.rows()) {
                return false;
            }
        }

        return true;
    }

    private List<Point> getSurroundingPoints(Point center) {
        return Arrays.asList(
                new Point(center.x - xScale, center.y - yScale),
                new Point(center.x, center.y - yScale),
                new Point(center.x + xScale, center.y - yScale),
                new Point(center.x - xScale, center.y),
                new Point(center.x, center.y),
                new Point(center.x + xScale, center.y),
                new Point(center.x - xScale, center.y + yScale),
                new Point(center.x, center.y + yScale),
                new Point(center.x + xScale, center.y + yScale)
        );
    }

    private int rowCount() {
        return BGR_REFERENCE_COLORS.size();
    }

    private int colCount() {
        return BGR_REFERENCE_COLORS.get(0).size();
    }
}