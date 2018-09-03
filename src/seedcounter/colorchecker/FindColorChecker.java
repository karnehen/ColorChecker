package seedcounter.colorchecker;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;


import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Feature2D;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import seedcounter.common.Quad;


public class FindColorChecker {
    private final Mat referenceImage;
    private final MatchingModel matchingModel;
    private final MatOfKeyPoint referenceKeypoints;
    private final MatOfKeyPoint referenceDescriptors;
    private final Feature2D detector;
    private final Feature2D extractor;

    public FindColorChecker(String referenceFile, MatchingModel matchingModel) {
        referenceImage = Imgcodecs.imread(referenceFile,
                Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR | Imgcodecs.CV_LOAD_IMAGE_ANYDEPTH);
        this.matchingModel = matchingModel;

        referenceKeypoints = new MatOfKeyPoint();
        detector = this.matchingModel.getDetector();
        detector.detect(referenceImage, referenceKeypoints);

        referenceDescriptors = new MatOfKeyPoint();
        extractor = this.matchingModel.getExtractor();
        extractor.compute(referenceImage, referenceKeypoints, referenceDescriptors);
    }

    public Quad findBestFitColorChecker(Mat image) {
        Quad bestQuad = findColorChecker(image);
        Mat extractedColorChecker = bestQuad.getTransformedField(image);
        ColorChecker colorChecker = new ColorChecker(extractedColorChecker,
                false, false);
        double bestMetric = colorChecker.labDeviationFromReference();
        extractedColorChecker.release();

        for (double scale : Arrays.asList(0.05, 0.1, 0.2)) {
            Quad quad = findColorChecker(image, scale);
            extractedColorChecker = quad.getTransformedField(image);
            colorChecker = new ColorChecker(extractedColorChecker,
                    false, false);
            double metric = colorChecker.labDeviationFromReference();
            extractedColorChecker.release();
            if (metric < bestMetric) {
                bestMetric = metric;
                bestQuad = quad;
            }
        }

        return bestQuad;
    }

    public Quad findColorChecker(Mat image) {
        return getQuad(getHomography(image), 0.0).orElse(fullImageQuad(image));
    }

    public Quad findColorChecker(Mat image, double scale) {
        Optional<Quad> quad1 = getQuad(getHomography(image), scale);
        if (!quad1.isPresent()) {
            return fullImageQuad(image);
        }

        image = imageSplice(image, quad1.get());
        Optional<Quad> quad2 = getQuad(getHomography(image), 0.0);
        image.release();
        if (!quad2.isPresent()) {
            return fullImageQuad(image);
        }

        return shiftQuad(quad2.get(), quad1.get());
    }

    private Quad fullImageQuad(Mat image) {
        return new Quad(
                new Point(0.0, 0.0),
                new Point(image.cols() - 1, 0.0),
                new Point(image.cols() - 1, image.rows() - 1),
                new Point(0.0, image.rows() - 1)
        );
    }

    private Optional<Mat> getHomography(Mat image) {
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        if (image.rows() < 100 || image.cols() < 50) {
            return Optional.empty();
        }
        detector.detect(image, keypoints);
        if ((int) keypoints.size().width * (int) keypoints.size().height < 2) {
            return Optional.empty();
        }

        MatOfKeyPoint descriptors = new MatOfKeyPoint();
        extractor.compute(image, keypoints, descriptors);

        LinkedList<DMatch> goodMatches = getGoodMatches(descriptors);

        if (goodMatches.isEmpty()) {
            return Optional.empty();
        }

        return getHomography(keypoints, goodMatches);
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

    private Optional<Mat> getHomography(MatOfKeyPoint keypoints, LinkedList<DMatch> goodMatches) {
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

        return Optional.of(Calib3d.findHomography(
                referenceMatOfPoint2f, matOfPoint2f, Calib3d.RANSAC, 3))
                .filter(x -> x.cols() > 0 && x.rows() > 0);
    }

    private Optional<Quad> getQuad(Optional<Mat> homography, double scale) {
        if (!homography.isPresent()) {
            return Optional.empty();
        }

        Mat corners = new Mat(4, 1, CvType.CV_32FC2);
        Mat referenceCorners = new Mat(4, 1, CvType.CV_32FC2);

        referenceCorners.put(0, 0, -scale * referenceImage.cols(), -scale * referenceImage.rows());
        referenceCorners.put(1, 0, (1.0 + scale) * referenceImage.cols(), -scale * referenceImage.rows());
        referenceCorners.put(2, 0, (1.0 + scale) * referenceImage.cols(), (1.0 + scale) * referenceImage.rows());
        referenceCorners.put(3, 0, -scale * referenceImage.cols(), (1.0 + scale) * referenceImage.rows());

        Core.perspectiveTransform(referenceCorners, corners, homography.get());

        return Optional.of(new Quad(new Point(corners.get(0, 0)),new Point(corners.get(1, 0)),
                new Point(corners.get(2, 0)), new Point(corners.get(3, 0))));
    }

    public void fillColorChecker(Mat image, Quad quad) {
        MatOfPoint points = new MatOfPoint();

        points.fromArray(quad.getPoints());
        Imgproc.fillConvexPoly(image, points, getBackgroundColor(image, quad));
    }

    private Mat imageSplice(Mat image, Quad quad) {
        Range rows = new Range(clipRow(top(quad), image), clipRow(bottom(quad), image) + 1);
        Range cols = new Range(clipCol(left(quad), image), clipCol(right(quad), image) + 1);

        return new Mat(image, rows, cols);
    }

    private int clipRow(int row, Mat image) {
        return Math.max(0, Math.min(image.rows() - 1, row));
    }

    private int clipCol(int col, Mat image) {
        return Math.max(0, Math.min(image.cols() - 1, col));
    }

    private Quad shiftQuad(Quad quad, Quad shift) {
        return new Quad(
                shiftPoint(quad.tl(), shift),
                shiftPoint(quad.tr(), shift),
                shiftPoint(quad.br(), shift),
                shiftPoint(quad.bl(), shift)
        );
    }

    private Point shiftPoint(Point point, Quad shift) {
        return new Point(point.x + left(shift), point.y + top(shift));
    }

    private int left(Quad quad) {
        return (int) Math.min(
                Math.min(quad.tl().x, quad.tr().x),
                Math.min(quad.bl().x, quad.br().x)
        );
    }

    private int right(Quad quad) {
        return (int) Math.max(
                Math.max(quad.tl().x, quad.tr().x),
                Math.max(quad.bl().x, quad.br().x)
        );
    }

    private int top(Quad quad) {
        return (int) Math.min(
                Math.min(quad.tl().y, quad.tr().y),
                Math.min(quad.bl().y, quad.br().y)
        );
    }

    private int bottom(Quad quad) {
        return (int) Math.max(
                Math.max(quad.tl().y, quad.tr().y),
                Math.max(quad.bl().y, quad.br().y)
        );
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
