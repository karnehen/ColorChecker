package seedcounter;

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
}
