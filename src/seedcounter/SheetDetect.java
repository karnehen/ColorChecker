package seedcounter;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;


public class SheetDetect {
	// ������� �� ���� ��������� �����������
	protected int heightPadding = 0;
	protected int widthPadding = 0;
	
	protected double MIN_SIDE_OF_QUAD = 0.33; // ����������� ������ ���������������� ������������ ������ ����������� (image.rows()) 
	protected static double NEAR_ANGLE = Math.PI/90;
	protected static double RIGHT_ANGLE = 1.5707963; // 90* in radians
	
	protected double AVERAGE_FACTOR = 0.001;
	
	protected double nearLineDistance;
	
	// ������� ��������� �� ������� ������� ����������� � ����������� ������������� �����
	protected int blockSize = 25;
	protected double C = 17;
	protected int medianBlur = 0;
	
	public SheetDetect(int height, int width) {
		nearLineDistance = width*0.023;
	
		double dpmm = width*height/(210*297*(5/3));
		
		if(dpmm<0.340339907) { //~	17649px(to 26473) 0.22671156004 dpmm	
			blockSize = 3;
			C = 15;
			medianBlur = 0;
		} else 
		if(dpmm<1.362089145) { //~ 70818px (to 106227)		0.91008497675 dpmm
			blockSize = 5;	
			C = 7;
			medianBlur = 3;
		} else
		if(dpmm<5.436050986) { //~ 282828px (to 424242)		3.62486772487 dpmm
			blockSize = 7;
			C = 17;
			medianBlur = 3;
		} else
		if(dpmm<21.79067661) { //~ 1132200px (to 1698300)		14.5303992304 dpmm
			blockSize = 35; 
			C = 15;
			medianBlur = 5;
		} else
		if(dpmm<87.08225108) { // >~ 4524475px (to 6786712)		58.0338624339 dpmm
			blockSize = 19;
			C = 3;
			medianBlur = 31;
		} else
		if(dpmm<348.7240019) { // >~ 18106549px (to 27159823)		232.486387686 dpmm
			blockSize = 235;
			C = 31;
			medianBlur = 35;
		} else { // >~ 72419094 px  				 930.111992945 dpmm
			blockSize = 71;
			C = 27;
			medianBlur = 151;
		}
	}
	
	
    /**
     * �������� ������� ����������� ������������ ����������������� quad, ������� ��������������� �������� ��������������.
     * @param image - �������� �����������.
     * @param quad - ���������������, ����������� ��������� ������� �����������. 
     * @return ��������� ������� ����� ��������������� ��������������.
     */
    public Mat getTransformedField(Mat image, Quad quad) {    	
    	// Define the destination image
    	Mat transformed = new Mat(image.rows(), image.cols(), image.type());
    	
    	// Corners of the destination image
    	Point[] quad_pts = new Point[4];
    	quad_pts[0] = new Point(0, 0);
    	quad_pts[1] = new Point(transformed.cols(), 0);
    	quad_pts[2] = new Point(transformed.cols(), transformed.rows());
    	quad_pts[3] = new Point(0, transformed.rows());

    	// Get transformation matrix
    	Mat transmtx = Imgproc.getPerspectiveTransform(new MatOfPoint2f(quad.getPoints()), new MatOfPoint2f(quad_pts));

    	// Apply perspective transformation
    	Imgproc.warpPerspective(image, transformed, transmtx, transformed.size());    	
    	transformed = transformed.submat(heightPadding, transformed.rows()-heightPadding, 
    			widthPadding, transformed.cols()-widthPadding);
    	
    	return transformed;
    }
	
	
	
	/**
     * ����������� ����� ������ �� ������ ����. 
     * @param image - �������� �����������.
     * @return ��������� ������ Quad, �������������� ���������������, ��������������� ����������� ������� ����� ������.
     */
    public Quad detectQuad(Mat image) {
    	// �������� ����������� ��� ���������� �����
		Mat blured = new Mat(image.rows(), image.cols(), CvType.CV_8UC3);
		Imgproc.medianBlur(image, blured, (int)medianBlur);
		
    	// �������������� ����������� � ������� ������
    	Mat gray = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);  	
    	Imgproc.cvtColor(blured, gray, Imgproc.COLOR_RGBA2GRAY);
    	blured.release();
    	
    	// ���������� �����������
    	Mat hulled = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);
    	Imgproc.adaptiveThreshold(gray, hulled, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, blockSize, C);
    	gray.release();
    	
    	// ����������� ����� ������ ����� ������
    	Mat lines = detectLines(hulled);
    	
    	// ����� ������� �����, ��������������� �������� �����
    	lines = findRectLines(lines, image.rows()*MIN_SIDE_OF_QUAD);
    	if(null == lines) return null;
    	
    	// ������������� ���������������� ���������������� ������� ����� ������
    	Quad quad = getQuad(lines);
    	
    	return quad;
    }
    
    
    /**
     * ������������� ���������������� ���������������� ������� ����� ������.
     * @param lines - �����, ������� �� �������� ����� ������.
     * @return ���������������, �������� ������ ����� ������.
     */
    private static Quad getQuad(Mat lines) {
    	Point[] corners = new Point[4];
    	 
    	// ����������� ������� ������������ �����
    	if(nearAngle(lines.get(0, 0), lines.get(0, 2))) {
    		corners[0] = computeIntersectPoint(lines.get(0, 0), lines.get(0, 1));
    		corners[1] = computeIntersectPoint(lines.get(0, 1), lines.get(0, 2));
    		corners[2] = computeIntersectPoint(lines.get(0, 2), lines.get(0, 3));
    		corners[3] = computeIntersectPoint(lines.get(0, 3), lines.get(0, 0));
    	} else 
    	if (nearAngle(lines.get(0, 0), lines.get(0, 1))) {
    		corners[0] = computeIntersectPoint(lines.get(0, 0), lines.get(0, 2));
    		corners[1] = computeIntersectPoint(lines.get(0, 2), lines.get(0, 1));
    		corners[2] = computeIntersectPoint(lines.get(0, 1), lines.get(0, 3));
    		corners[3] = computeIntersectPoint(lines.get(0, 3), lines.get(0, 0));
    	} else {
    		corners[0] = computeIntersectPoint(lines.get(0, 0), lines.get(0, 2));
    		corners[1] = computeIntersectPoint(lines.get(0, 2), lines.get(0, 3));
    		corners[2] = computeIntersectPoint(lines.get(0, 3), lines.get(0, 1));
    		corners[3] = computeIntersectPoint(lines.get(0, 1), lines.get(0, 0));
    	}
    	
    	// ���������� �����-�������
    	sortQuadCorners(corners);
    	
    	Quad quad = new Quad(corners[0], corners[1], corners[2], corners[3]);
    	return quad;
    }
    
    
    /**
     * ��������� ����� ����������� ���� ������.
     * @param line1 ������ �������� �������� (x1, y1)<->(x2, y2).
     * @param line2 ������ �������� �������� (x1, y1)<->(x2, y2).
     * @return - ����� ����������� ���� ������ (Point).
     */
    public static Point computeIntersectPoint(double[] line1, double[] line2) {
        return new Point(computeIntersect(line1, line2));
    }
    
    
    /**
     * ��������� ����� ����������� ���� ������.
     * @param line1 ������ �������� �������� (x1, y1)<->(x2, y2).
     * @param line2 ������ �������� �������� (x1, y1)<->(x2, y2).
     * @return - ����� ����������� ���� ������ (double[]).
     */
    public static double[] computeIntersect(double[] line1, double[] line2) {
        double x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
        double x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];

        double d = ((x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4));
        if (0 != d) {
            double[] pt = new double[2];
            pt[0] = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d;
            pt[1] = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d;
            return pt;
        } else
            return new double[]{-1, -1};
    }
    
    
    /**
     * ������������� ����� ���������������� ��������� �������: ������� ����� ����, ������ ������ ����, ������ ����� ����, ������ ������.
     * @param corners ����������������� ������ �������.
     * @return ��������������� ������ �������.
     */
	protected static boolean sortQuadCorners(Point[] corners) {	
		// Get mass center
		Point center = new Point(0,0);
		for (int i=0; i<corners.length; i++) {
			center.x += corners[i].x;
			center.y += corners[i].y;
		}
		
		center.x *= (1.0 / corners.length);
		center.y *= (1.0 / corners.length);
		
		
		List<Point> top = new ArrayList<Point>();
		List<Point> bot = new ArrayList<Point>();
		for (int i=0; i<corners.length; i++) {
			if (corners[i].y <= center.y)
				top.add(corners[i]);
			else
				bot.add(corners[i]);
		}
		if(top.size()!=2 || bot.size()!=2)
			return false;
			
		Point tl = top.get(0).x > top.get(1).x ? top.get(1) : top.get(0);
		Point tr = top.get(0).x > top.get(1).x ? top.get(0) : top.get(1);
		Point bl = bot.get(0).x > bot.get(1).x ? bot.get(1) : bot.get(0);
		Point br = bot.get(0).x > bot.get(1).x ? bot.get(0) : bot.get(1);
		
		corners[0] = tl;
		corners[1] = tr;
		corners[2] = br;
		corners[3] = bl;
			    
		return true;
	}
    
    
	/**
	 * �������� �� ���� ����� ����� ������� �����?
	 * @param line1 ������ �����.
	 * @param line2 ������ �����.
	 * @return true - ���� ����� (������������ ��������� NEAR_ANGLE), false - ������� ����.
	 */
    public static boolean nearAngle(double[] line1, double[] line2) {
    	double angleDiff = Math.abs(angle(line1, line2));
    	if(angleDiff>NEAR_ANGLE)
    		return false;
    	else 
    		return true;
    }
    
    
    /**
     * ���������� ���� ����� ����� �������.
     * @param line1 ������ �������� �������� (x1, y1)<->(x2, y2).
     * @param line2 ������ �������� �������� (x1, y1)<->(x2, y2).
     * @return ���� ����� �������.
     */
    public static double angle(double[] line1, double[] line2) {
    	double vect1X = Math.abs(line1[0]-line1[2]); // |x1 - x2|
		double vect1Y = Math.abs(line1[1]-line1[3]); // |y1 - y2|
		double length1 = vectLenght(vect1X, vect1Y);
		
		double vect2X = Math.abs(line2[0]-line2[2]); // |x1 - x2|
		double vect2Y = Math.abs(line2[1]-line2[3]); // |y1 - y2|
		double length2 = vectLenght(vect2X, vect2Y); 
		
		return Math.acos((vect1X*vect2X+vect1Y*vect2Y)/(length1*length2)); // angle
    }
    
    
    /**
     * ��������� ����� �������.
     * @param x ���������� �������.
     * @param y ���������� �������.
     * @return ����� �������.
     */
    public static double vectLenght(double x, double y) {
    	return Math.sqrt(x*x+y*y);
    }
    
    
    /**
     * ���������� ����� (�������) �� �����������, ��������������� �������� ����� ������.
     * @param image �������������� �����������.
     * @return ������������ ����� (�������).
     */
    public Mat detectLines(Mat image) {
    	double minLineLength = (int) (image.cols()*0.33);
    	double maxLineGap= (int) (image.cols()*0.21);
    	
    	Mat lines = new Mat();
    	Imgproc.HoughLinesP(image, lines, 1, Math.PI/180, 100, minLineLength, maxLineGap);
    	
    	return lines;
    }
    
    
    /**
     * ����� ����� �� ������, ����������� ���������������.
     * @param lines ��� �����.
     * @return �����, ����������� ���������������.
     */
    public Mat findRectLines(Mat lines, double minDistanceBetweenParallel) {
    	// glue near lines
    	List<double[]> gluedLines = glueLines(lines);
    	if(gluedLines.size()<4) return null;
    	
    	// list of group (per four line) of perhaps rectangle
    	List<List<double[]>> rectangles = new LinkedList<List<double[]>>();
    	for(double[] line : gluedLines) {
    		// list of: two max athwart and one max parallel line to current line
    		List<double[]> rect = new LinkedList<double[]>(); // current rectangle
    		List<double[]> forSeach = new LinkedList<double[]>();
    		for(double[] cp : gluedLines) forSeach.add(cp);
    		
    		// select bests lines for current
    		forSeach.remove(line);
    		double[] maxAthwart1 = getMaxAthwart(line, forSeach);
    		forSeach.remove(maxAthwart1);
    		double[] maxAthwart2 = getMaxAthwart(line, forSeach);
    		forSeach.remove(maxAthwart2);
    		
    		// for parallel exist criterion: minimal distance to it
    		double[] maxParallel = null;
    		while(0!=forSeach.size()) {
    			double[] current = getMaxParallel(line, forSeach);
    			
    			if(maxDistanceBetweenLineCorners(current, line) > minDistanceBetweenParallel) {
    				maxParallel = current;
    				break;
    			}
    			
    			forSeach.remove(current);
    		}
    		// if criterion isn't completed
    		if(null == maxParallel) return null;
    		
    		//double[] maxParallel = getMaxParallel(line, forSeach);
    		
    		rect.add(line); // current line
    		rect.add(maxAthwart1); // first best athwart line to current line
    		rect.add(maxAthwart2); // second best athwart line to current line
    		rect.add(maxParallel); // best parallel line to current line
    		rect.add(new double[]{										// deviation:
    					Math.abs(angle(line, maxAthwart1)-RIGHT_ANGLE), // from 90, between first athwart and current line 
    					Math.abs(angle(line, maxAthwart2)-RIGHT_ANGLE), // from 90, between second athwart and current line
    					angle(line, maxParallel)}); // from 0, between parallel and current line.
    		rectangles.add(rect);
    	}
    	
    	// find best rectangle in list of all
    	List<double[]> best = null;
    	double bestKoef = 99999;
    	for(List<double[]> current : rectangles) {
    		double curKoef = current.get(4)[0]+current.get(4)[1]+current.get(4)[2];
    		if(curKoef<bestKoef) {
    			best = current;
    			bestKoef = curKoef;
    		}
    			
    	}
    	if(null==best) return null;
    	
    	// copy best rectangle
    	Mat rectLines = new Mat(1, 4, lines.type());
    	for(int i=0; i<4; i++)
			rectLines.put(0, i, best.get(i));
    	
    	return rectLines;
    }
    
    
    /**
     * ��������� ��������� ����� �������� ���������� ������� ���� ��������.
     * @param line1 ������ �������.
     * @param line2 ������ �������.
     * @return ���������� ����� �������� ���������� ������� ���� ��������
     */
    public static double maxDistanceBetweenLineCorners(double[] line1, double[] line2) {
    	double max = 0;
    	double current = 0;
    	
    	// A1 to B1:	sqrt( (A1x-B1x)(A1x-B1x)+(A1y-B1y)(A1y-B1y) )
    	current = Math.sqrt((line1[0]-line2[0])*(line1[0]-line2[0])+(line1[1]-line2[1])*(line1[1]-line2[1]));
    	if(current>max)
    		max = current;
    	
    	// A2 to B2:	sqrt( (A2x-B2x)(A2x-B2x)+(A2y-B2y)(A2y-B2y) )
    	current = Math.sqrt((line1[2]-line2[2])*(line1[2]-line2[2])+(line1[3]-line2[3])*(line1[3]-line2[3]));
    	if(current>max)
    		max = current;
    	
    	// A1 to B2:	sqrt( (A1x-B2x)(A1x-B2x)+(A1y-B2y)(A1y-B2y) )
    	current = Math.sqrt((line1[0]-line2[2])*(line1[0]-line2[2])+(line1[1]-line2[3])*(line1[1]-line2[3]));
    	if(current>max)
    		max = current;
    	
    	// A2 to B1:	sqrt( (A2x-B1x)(A2x-B1x)+(A2y-B1y)(A2y-B1y) )
    	current = Math.sqrt((line1[2]-line2[0])*(line1[2]-line2[0])+(line1[3]-line2[1])*(line1[3]-line2[1]));
    	if(current>max)
    		max = current;
    	
    	return max;
    }
    
    
    /**
     * ������� �� ������ ������, �������� ���������, �������� ������������ ������.
     * @param line ������ ������.
     * @param lines ������ ������ ��� ������.
     * @return ������, �������� ������������ ������.
     */
    private static double[] getMaxParallel(double[] line, List<double[]> lines) {
    	double bestAngle = Double.MAX_VALUE;
    	double[] bestLine = null;
    	
    	for(double[] current : lines) {
			double newAngle = angle(line, current);
			if(newAngle<bestAngle) {
				bestLine = current;
				bestAngle = newAngle;
    		}
    	}
    	
    	return bestLine;
    }
    
    
    /**
     * ������� �� ������ ������, �������� ���������, �������� ���������������� ������.
     * @param line ������ ������.
     * @param lines ������ ������ ��� ������.
     * @return ������, �������� ���������������� ������.
     */
    private static double[] getMaxAthwart(double[] line, List<double[]> lines) {
    	double bestAngle = Double.MAX_VALUE;
    	double[] bestLine = null;
    	
    	for(double[] current : lines) {
			double newAngle = Math.abs(Math.abs(angle(line, current)) - RIGHT_ANGLE);
			if(newAngle<bestAngle) {
				bestLine = current;
				bestAngle = newAngle;
    		}
    	}
    	
    	return bestLine;
    }
    
    
    /**
     * "�������" ������� � ������ �����, ������������� �� �� ���������� "������".
     * @param lines ����� �����.
     * @return ������ "���������" �����.
     */
    protected List<double[]> glueLines(Mat lines) {
    	List<double[]> classifiedLines = new LinkedList<double[]>();
    	double[] frequences = new double[lines.cols()];

    	for(int i=0; i<lines.cols(); i++) {
    		boolean matched = false;
    		double[] nextLine = lines.get(0, i); // x1, y1, x2, y2		
    		
    		for(int j=0; j<classifiedLines.size(); j++) {
    			double[] classifiedLine = classifiedLines.get(j);
    			
    			if(nearLines(nextLine, classifiedLine)) {
    				matched = true;
    				average(classifiedLine, nextLine);
    				frequences[j]++;
    			}
    		}
    		
    		// adding new line class
    		if(!matched) {
    			frequences[classifiedLines.size()]++;
    			classifiedLines.add(nextLine);
    		}
    	}
    	
    	sortLinesByFrequency(classifiedLines, frequences);
    	
    	return classifiedLines;
    }
    
    
	/**
     * ������������� ������ ����� �� ������� "������".
     * @param ������ �����.
     * @param ������� "������".
     */
    private static void sortLinesByFrequency(final List<double[]> classifiedLines, final double[] frequences) {
    	Collections.sort(classifiedLines, new Comparator<double[]>() {
			@Override
			public int compare(double[] lhs, double[] rhs) {
				double lfr = frequences[classifiedLines.indexOf(lhs)];
				double rfr = frequences[classifiedLines.indexOf(rhs)];
				
				if(lfr<rfr)
					return 1;
				if(lfr>rfr)
					return -1;
				else
					return 0;
			}
		});
    }
    
    
    /**
     * ����������, �������� �� ��� ����� �������� � �������.
     * @param line1 ������ �����.
     * @param line2 ������ �����.
     * @return true ����� �����, false �����.
     */
    public boolean nearLines(double[] line1, double[] line2) {
    	// Line1:
    	double x1 = line1[0], y1 = line1[1]; // P1  
    	double x2 = line1[2], y2 = line1[3]; // P2
    	
    	//  Line2:
    	double x11 = line2[0], y11 = line2[1]; // P11
    	double x22 = line2[2], y22 = line2[3]; // P22
    	
    	if(!nearAngle(line1, line2))
    		return false;
    	
    	// For ends of segments is near
    	if(Math.abs(x1-x11)<nearLineDistance && Math.abs(y1-y11)<nearLineDistance)
    		return true;
    	
    	if(Math.abs(x2-x22)<nearLineDistance && Math.abs(y2-y22)<nearLineDistance)
    		return true;
    	
    	if(Math.abs(x1-x22)<nearLineDistance && Math.abs(y1-y22)<nearLineDistance)
    		return true;
    	
    	if(Math.abs(x2-x11)<nearLineDistance && Math.abs(y2-y11)<nearLineDistance)
    		return true;
    	
    	// One of segment on other line
    	if(isNearPointToLine(new double[]{x1, y1}, line2)) //if(p1->line2) 
    		if(isNearPointToLine(new double[]{x2, y2}, line2) || // if(p2 -> line2 || (p11 -> line1 || p22 ->line1 ))
    				(isNearPointToLine(new double[]{x11, y11}, line1) || isNearPointToLine(new double[]{x22, y22}, line1)))
    			return true;
    		
    	if(isNearPointToLine(new double[]{x11, y11}, line1)) //if(p11 -> line1) 
    		if(isNearPointToLine(new double[]{x22, y22}, line1) || // if(p22 -> line1 || (p1 -> line2 || p2 ->line2 ))
    				(isNearPointToLine(new double[]{x1, y1}, line2) || isNearPointToLine(new double[]{x2, y2}, line2)))
    			return true;
    	
    	return false;
    }
    
    
    /**
     * ����������, ��������� �� ������ ����� ������ � ������.
     * @param point - ������ �����.
     * @param line - ������.
     * @return true ���� ���������� �� ����� �� ������ �����, false �����.
     */
    public boolean isNearPointToLine(double[] point, double[] line) {
    	double distance = distanceFromPointToLine(point, line);
    	return distance<(nearLineDistance/2);
    }
    
    public void average(double[] line1, double[] line2) {
		// past together
		if(lineLenght(line1)>lineLenght(line2)) {
			line1[0] = line1[0]*(1-AVERAGE_FACTOR)+line2[0]*AVERAGE_FACTOR;
			line1[1] = line1[1]*(1-AVERAGE_FACTOR)+line2[1]*AVERAGE_FACTOR;
			line1[2] = line1[2]*(1-AVERAGE_FACTOR)+line2[2]*AVERAGE_FACTOR;
			line1[3] = line1[3]*(1-AVERAGE_FACTOR)+line2[3]*AVERAGE_FACTOR;
		} else {
			line1[0] = line2[0]*(1-AVERAGE_FACTOR)+line1[0]*AVERAGE_FACTOR;
			line1[1] = line2[1]*(1-AVERAGE_FACTOR)+line1[1]*AVERAGE_FACTOR;
			line1[2] = line2[2]*(1-AVERAGE_FACTOR)+line1[2]*AVERAGE_FACTOR;
			line1[3] = line2[3]*(1-AVERAGE_FACTOR)+line1[3]*AVERAGE_FACTOR;
		}
    }
    
    
    /**
     * ���������� �� ����� �� ������.
     * @param point - �����.
     * @param line - ������.
     * @return ���������� �� ����� �� ������.
     */
    public static double distanceFromPointToLine(double[] point, double[] line) {
		double A = (line[3]-line[1]);
		double B = -(line[2]-line[0]);
		double C = (line[1]*line[2] - line[1]*line[0] - line[0]*line[3] + line[0]*line[1]);
		
		double tmp = Math.sqrt(A*A + B*B);
		if(0!=tmp) return Math.abs(A*point[0] + B*point[1] + C) / tmp;
		return distance(point, new double[]{line[0], line[1]});
    }
    
    
    /**
     * ���������� ����� ����� �������.
     * @param p1 - ������ �����.
     * @param p2 - ������ �����.
     * @return ���������� ����� ����� �������.
     */
    public static double distance(double[] p1, double[] p2) {
    	double vectX = Math.abs(p2[0]-p1[0]);
    	double vectY = Math.abs(p2[1]-p1[1]);
    	return Math.sqrt((vectX*vectX) + (vectY*vectY));
    }
    
    
    /**
     * ����� �������.
     * @param line - �������, �������� ������������: x1, y1, x2, y2. 
     * @return - ����� �������.
     */
    public static double lineLenght(double[] line) {
    	double vectX = Math.abs(line[0]-line[2]);
    	double vectY = Math.abs(line[1]-line[3]);
    	return Math.sqrt(vectX*vectX+vectY*vectY);
    }
}
