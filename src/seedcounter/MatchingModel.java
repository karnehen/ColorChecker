package seedcounter;

public class MatchingModel {
	private final int detector;
	private final int extractor;
	private final int matcher;
	private final float threshold;

	public MatchingModel(int detector, int extractor,
			int matcher, float threshold) {
		this.detector = detector;
		this.extractor = extractor;
		this.matcher = matcher;
		this.threshold = threshold;
	}

	public int getDetector() {
		return detector;
	}

	public int getExtractor() {
		return extractor;
	}

	public int getMatcher() {
		return matcher;
	}

	public float getThreshold() {
		return threshold;
	}
}
