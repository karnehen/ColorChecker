package seedcounter;


import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

/**
 * Quad.
 * @author Komyshev
 * @version 0.2.
 */
public class Quad {
	private final Point tl;
	private final Point tr;
	private final Point br;
	private final Point bl;
	
	public Quad(Point tl, Point tr, Point br, Point bl) {
		this.tl = tl;
		this.tr = tr;
		this.br = br;
		this.bl = bl;
	}
	
	public Quad(Point tl, Point br) {
		this.tl = tl;
		this.br = br;
		
		this.tr = new Point(br.x, tl.y);
		this.bl = new Point(tl.x, br.y);
	}
	
	public Quad(double[][] points) {
		this.tl = new Point(points[0][0], points[0][1]);
		this.tr = new Point(points[1][0], points[1][1]);
		this.br = new Point(points[2][0], points[2][1]);
		this.bl = new Point(points[3][0], points[3][1]);
	}
	
	public Quad(double tlx, double tly, double trx, double tryy, double blx, double bly, double brx, double bry) {
		this.tl = new Point(tlx, tly);
		this.tr = new Point(trx, tryy);
		this.br = new Point(blx, bly);
		this.bl = new Point(brx, bry);
	}
	
	public Point[] getPoints() {
		return new Point[]{tl, tr, br, bl};
	}
	
	public double getBigSideSize() {
		if(Math.abs(tl.x-tr.x) > Math.abs(tl.y-bl.y)) {
			return tl.x-tr.x;
		} else {
			return tl.y-bl.y;
		}
	}
	
	public double getSmallSideSize() {
		if(Math.abs(tl.x-tr.x) > Math.abs(tl.y-bl.y)) {
			return tl.y-bl.y;
		} else {
			return tl.x-tr.x;
		}
	}

	// TODO: implement precise version
	public boolean isInside(Point point) {
		double minX = Math.min(Math.min(tl.x, tr.x), Math.min(bl.x, br.x));
		double minY = Math.min(Math.min(tl.y, tr.y), Math.min(bl.y, br.y));
		double maxX = Math.max(Math.max(tl.x, tr.x), Math.max(bl.x, br.x));
		double maxY = Math.max(Math.max(tl.y, tr.y), Math.max(bl.y, br.y));
		return point.x >= minX && point.x <= maxX && point.y >= minY && point.y <= maxY;
	}

	public double getArea() {
		double dx1 = tr.x - tl.x;
		double dy1 = tr.y - tl.y;
		double dx2 = bl.x - tl.x;
		double dy2 = bl.y - tl.y;

		return Math.abs(dx1 * dy2 - dy1 * dx2);
	}
	
	public Point tl() {
		return tl;
	}
	
	public Point tr() {
		return tr;
	}
	
	public Point bl() {
		return bl;
	}
	
	public Point br() {
		return br;
	}
	
	/**
	 * Recalculate relational points coordinates at source image for distinct image. The result is quad that has corresponding coordinates in distinct image. 
	 * @param from is the source image.
	 * @param to is the distinct image.
	 * @param srcQuad is the quad for recalculate. 
	 * @return recalculated quad.
	 */
	public static Quad recalculateQuadPoints(Mat from, Mat to, Quad srcQuad) {
		int width = from.cols();
		int height = from.rows();
		
		int newWidth = to.cols();
		int newHeight = to.rows();
		
		Point newTl = new Point(srcQuad.tl().x/width*newWidth, srcQuad.tl().y/height*newHeight);
		Point newTr = new Point(srcQuad.tr().x/width*newWidth, srcQuad.tr().y/height*newHeight);
		Point newBl = new Point(srcQuad.bl().x/width*newWidth, srcQuad.bl().y/height*newHeight);
		Point newBr = new Point(srcQuad.br().x/width*newWidth, srcQuad.br().y/height*newHeight);
		
		return new Quad(newTl, newTr, newBr, newBl);
	}

    public Mat getTransformedField(Mat image) {    	
    	// Define the destination image
    	Mat transformed = new Mat(image.rows(), image.cols(), image.type());
    	
    	// Corners of the destination image
    	Point[] quad_pts = new Point[4];
    	quad_pts[0] = new Point(0, 0);
    	quad_pts[1] = new Point(transformed.cols(), 0);
    	quad_pts[2] = new Point(transformed.cols(), transformed.rows());
    	quad_pts[3] = new Point(0, transformed.rows());

    	// Get transformation matrix
    	Mat transmtx = Imgproc.getPerspectiveTransform(new MatOfPoint2f(getPoints()),
    			new MatOfPoint2f(quad_pts));

    	// Apply perspective transformation
    	Imgproc.warpPerspective(image, transformed, transmtx, transformed.size());
    	
    	return transformed;
    }
}