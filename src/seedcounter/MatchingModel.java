package seedcounter;

import org.opencv.features2d.Feature2D;

public class MatchingModel {
    private final Feature2D detector;
    private final Feature2D extractor;
    private final int matcher;
    private final float threshold;

    public MatchingModel(Feature2D detector, Feature2D extractor,
            int matcher, float threshold) {
        this.detector = detector;
        this.extractor = extractor;
        this.matcher = matcher;
        this.threshold = threshold;
    }

    public Feature2D getDetector() {
        return detector;
    }

    public Feature2D getExtractor() {
        return extractor;
    }

    public int getMatcher() {
        return matcher;
    }

    public float getThreshold() {
        return threshold;
    }
}
