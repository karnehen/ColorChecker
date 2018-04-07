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
    private final Integer yScale;

    private static final Double REAL_WIDTH = 64.0; // millimeters
    private static final Double REAL_HEIGHT = 108.0; // millimeters

    public ColorChecker(Mat image) {
        this.checkerImage = image;
        Integer width = image.width();
        Integer height = image.height();

        xScale = (int) (0.04 * width);
        yScale = (int) (0.02 * height);
        List<Double> xCenters = Arrays.asList(0.143, 0.381, 0.613, 0.862);
        List<Double> yCenters = Arrays.asList(0.160, 0.305, 0.440, 0.580, 0.717, 0.856);

        this.centers = new ArrayList<>();
        for (Double y : yCenters) {
            List<Point> points = new ArrayList<>();
            for (Double x : xCenters) {
                points.add(new Point(x * width, y * height));
            }
            this.centers.add(points);
        }
    }

    public Double pixelArea(Quad quad) {
        return REAL_WIDTH * REAL_HEIGHT / quad.getArea();
    }

    public Mat calibrate(Mat srcImage, RegressionModel model,
                         ColorSpace featuresSpace, ColorSpace targetSpace) {
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

        for (Integer row = 0; row < BGR_REFERENCE_COLORS.size(); ++row) {
            for (Integer col = 0; col < BGR_REFERENCE_COLORS.get(0).size(); ++col) {
                List<DoubleBuffer> samplePoints = getSamplePoints(row, col);
                for (DoubleBuffer s : samplePoints) {
                    train.add(featuresSpace.convertFromBGR(s, false));
                    DoubleBuffer referenceColor = DoubleBuffer.wrap(BGR_REFERENCE_COLORS.get(row).get(col).val);
                    answers.add(targetSpace.convertFromBGR(referenceColor, false));
                }
            }
        }

        try {
            model.train(train, answers);
        } catch (SingularMatrixException e) {
            return result;
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

    public CellColors getCellColors(Mat checkerImage) {
        CellColors cellColors = new CellColors();

        for (Integer row = 0; row < 6; ++row) {
            for (Integer col = 0; col < 4; ++col) {
                List<DoubleBuffer> actualColors = getSamplePoints(checkerImage, row, col, true);
                DoubleBuffer referenceColor = DoubleBuffer.wrap(BGR_REFERENCE_COLORS.get(row).get(col).val);
                for (DoubleBuffer color : actualColors) {
                    cellColors.addColor(new Color(color), new Color(referenceColor));
                }
            }
        }

        return cellColors;
    }

    private List<DoubleBuffer> getSamplePoints(Integer row, Integer col) {
        return getSamplePoints(this.checkerImage, row, col, false);
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

        for (Integer row = 0; row < 6; ++row) {
            for (Integer col = 0; col < 4; ++col) {
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

    private List<Point> getSurroundingPoints(Point center) {
        return Arrays.asList(
                new Point(center.x - this.xScale, center.y - this.yScale),
                new Point(center.x, center.y - this.yScale),
                new Point(center.x + this.xScale, center.y - this.yScale),
                new Point(center.x - this.xScale, center.y),
                new Point(center.x, center.y),
                new Point(center.x + this.xScale, center.y),
                new Point(center.x - this.xScale, center.y + this.yScale),
                new Point(center.x, center.y + this.yScale),
                new Point(center.x + this.xScale, center.y + this.yScale)
        );
    }
}