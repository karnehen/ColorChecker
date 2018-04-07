package seedcounter.common;

import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import javafx.util.Pair;

public class HSVBinarization {
    private final List<Pair<Scalar, Scalar>> targetsAndRanges;

    public HSVBinarization(List<Pair<Scalar, Scalar>> targetsAndRanges) {
        this.targetsAndRanges = targetsAndRanges;
    }


    public Mat apply(Mat input) {
        int width = input.cols();
        int height = input.rows();

        Mat hsvImg = new Mat(height, width, CvType.CV_8UC3);
        Imgproc.cvtColor(input, hsvImg, Imgproc.COLOR_BGR2HSV);

        Mat output = new Mat(height, width, CvType.CV_8UC1);
        for(int i=0; i<hsvImg.rows(); i++) {
            for(int j=0; j<hsvImg.cols(); j++) {
                double[] value = hsvImg.get(i, j);

                boolean hitting = false;
                for(Pair<Scalar, Scalar> targetAndRange : targetsAndRanges) {
                    Scalar target = targetAndRange.getKey();
                    Scalar range = targetAndRange.getValue();

                    if(Math.abs(value[0]-target.val[0])<range.val[0] && Math.abs(value[1]-target.val[1])<range.val[1] && Math.abs(value[2]-target.val[2])<range.val[2])
                        hitting = true;
                }

                if(hitting) {
                    output.put(i, j, 255);
                } else {
                    output.put(i, j, 0);
                }
            }
        }

        return output;
    }

}
