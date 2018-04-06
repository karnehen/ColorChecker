package seedcounter;

import java.nio.DoubleBuffer;

public enum ColorSpace {
    RGB(false, false),
    RGB_LINEAR(false, true),
    XYZ(true, false),
    XYZ_LINEAR(true, true);

    private boolean isXYZ;
    private boolean isLinear;

    ColorSpace(boolean isXYZ, boolean isLinear) {
        this.isXYZ = isXYZ;
        this.isLinear = isLinear;
    }

    public boolean isXYZ() {
        return isXYZ;
    }

    public boolean isLinear() {
        return isLinear;
    }

    DoubleBuffer convertFromBGR(DoubleBuffer bgr) {
        double b = Color.channel(bgr, 0);
        double g = Color.channel(bgr, 1);
        double r = Color.channel(bgr, 2);

        if (isLinear) {
            b = Color.linearizeRGB(b);
            g = Color.linearizeRGB(g);
            r = Color.linearizeRGB(r);
        }
        double[] result = {0.0, 0.0, 0.0};

        if (isXYZ) {
            result[0] = r * 0.4124
                    + g * 0.3576
                    + b * 0.1805;
            result[1] = r * 0.2126
                    + g * 0.7152
                    + b * 0.0722;
            result[2] = r * 0.0193
                    + g * 0.1192
                    + b * 0.9505;
        } else {
            result[0] = b;
            result[1] = g;
            result[2] = r;
        }

        return DoubleBuffer.wrap(result);
    }
}
