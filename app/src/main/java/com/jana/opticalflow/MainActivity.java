package com.jana.opticalflow;

import android.Manifest;
import android.content.pm.PackageManager;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.WindowManager;
import java.util.ArrayList;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.GFTTDetector;

// import static org.opencv.video.Video.OPTFLOW_FARNEBACK_GAUSSIAN;
// import static org.opencv.video.Video.OPTFLOW_USE_INITIAL_FLOW;
// import static org.opencv.video.Video.calcOpticalFlowFarneback;
import java.util.ArrayList;
import java.util.Random;

import static org.opencv.video.Video.calcOpticalFlowPyrLK;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{
    private int mViewMode;
    private Mat mRgba;
    private Mat mGray;
    private Mat mPrevGray;


    private CameraBridgeViewBase cameraView;
    private Mat previousFrame;
    private MatOfPoint2f prevPts;
    private Point center;

    private long prevFrameTime;
    private ArrayList<Double> xvels;
    private ArrayList<Double> yvels;
    //private ArrayList<Double> xcenters;
    //private ArrayList<Double> ycenters;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        OpenCVLoader.initDebug();
        cameraView = findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    1);
        }
        else {
            cameraView.enableView();
        }
        prevPts = new MatOfPoint2f();
        center = new Point();
        xvels = new ArrayList<>();
        yvels = new ArrayList<>();
        //xcenters = new ArrayList<>();
        //ycenters = new ArrayList<>();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case 1: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    cameraView.enableView();
                }
                return;
            }
        }
    }

    public Point getCentroid(Point[] points){
        if(points.length == 0) {
            return new Point();
        }
        int sumx = 0;
        int sumy = 0;
        for(int i = 0; i<points.length;i++){
            sumx+=points[i].x;
            sumy+=points[i].y;
        }
        Point centroid = new Point(sumx/points.length, sumy/points.length);
        return centroid;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        long time = System.currentTimeMillis();
        double timeDiff = (time - prevFrameTime)/1000.;
        prevFrameTime = time;
        Mat gray = inputFrame.gray();
        Mat rgb = inputFrame.rgba();
        MatOfPoint2f nextPts = new MatOfPoint2f();
        /*MatOfPoint2f centerPts = new MatOfPoint2f();
        ArrayList<Point> centerList = new ArrayList<Point>();
        centerList.add(center);
        Random rand = new Random();
        for (int i=0; i<20; i++) {
            double rad = rand.nextDouble()*20;
            double rad1 = rand.nextDouble()*20;
            if (rad > 10) {
                rad = -rad/2;
            }
            if (rad1 > 10) {
                rad1 = -rad1/2;
            }
            double newx = center.x+rad;
            double newy = center.y+rad1;
            centerList.add(new Point(newx, newy));

        }
        centerPts.fromList(centerList);*/
        MatOfByte status = new MatOfByte();
        MatOfFloat err = new MatOfFloat();
        if (previousFrame != null) {
            calcOpticalFlowPyrLK(previousFrame, gray, prevPts, nextPts, status, err, new Size(21, 21), 2);

            double totalx = 0;
            double totaly = 0;
            double totalvelx = 0;
            double totalvely = 0;
            double totaldisplacex = 0;
            double totaldisplacey = 0;
            int numFound = 0;
            double maxdis = 0;
            int maxdisind = 0;
            //TODO:choose largest displacement
            for (int i = 0; i < nextPts.toList().size(); i++) {
                if (status.toList().get(i) == 1) {
                    totalx += nextPts.toArray()[i].x;
                    totaly += nextPts.toArray()[i].y;
                    Point displacement = prevPts.toArray()[i].subtract(nextPts.toArray()[i]);
                    double displacementmag = Math.sqrt(Math.pow(displacement.x, 2) + Math.pow(displacement.y, 2));
                    if (displacementmag > maxdis) {
                        maxdis = displacementmag;
                        maxdisind = i;
                    }
                    totalvelx += displacement.x / timeDiff;
                    totalvely += displacement.y / timeDiff;
                    totaldisplacex += displacement.x;
                    totaldisplacey += displacement.y;
                    numFound++;
                }
            }
            Point displacement = nextPts.toArray()[maxdisind].subtract(prevPts.toArray()[maxdisind]);
            double displacex = displacement.x;//totaldisplacex/numFound;
            double displacey = displacement.y;//totaldisplacey/numFound;
            double centerx = nextPts.toArray()[maxdisind].x;//totalx/numFound;
            double centery = nextPts.toArray()[maxdisind].y;//totaly/numFound;
            double xvelp = displacex / timeDiff;//totalvelx/numFound;
            double yvelp = displacey / timeDiff;//totalvely/numFound;
            double xvel = (3. / rgb.cols()) * xvelp;
            double yvel = (2. / rgb.rows()) * yvelp;
            if (xvels.size() >= 5) {
                xvels.remove(0);
                yvels.remove(0);
                //xcenters.remove(0);
                //ycenters.remove(0);
            }
            xvels.add(xvel);
            yvels.add(yvel);
            //xcenters.add(centerx);
            //ycenters.add(centery);
            xvel = 0;
            yvel = 0;
            //centerx = 0;
            //centery = 0;
            for (int i = 0; i < xvels.size(); i++) {
                xvel += xvels.get(i);
                yvel += yvels.get(i);
                //centerx += xcenters.get(i);
                //centery += ycenters.get(i);
            }
            xvel = xvel / 5.;
            yvel = yvel / 5.;
            //centerx = centerx/5.;
            //centery = centery/5.;

            double magn = Math.sqrt(Math.pow(xvel, 2) + Math.pow(yvel, 2));
            if (magn > 0.05) {
                Imgproc.arrowedLine(rgb, new Point(centerx, centery), new Point(centerx + displacex, centery + displacey), new Scalar(255, 0, 0), 10);
            } else {
                xvel = 0;
                yvel = 0;
            }
            Imgproc.putText(rgb, "XVEL" + String.valueOf(xvel) + "m/s", new Point(30, 50), Core.FONT_HERSHEY_PLAIN, 5.0, new Scalar(0, 0, 255));
            Imgproc.putText(rgb, "YVEL" + String.valueOf(yvel) + "m/s", new Point(30, 100), Core.FONT_HERSHEY_PLAIN, 5.0, new Scalar(0, 0, 255));
        }

        previousFrame = inputFrame.gray();
        // MatofPoint object for import features (corners)
        MatOfPoint points = new MatOfPoint();

        // Generating an object of the most important features (we are using corners of objects)
        Imgproc.goodFeaturesToTrack(previousFrame, points, 100, .1, 21, new Mat(), 3, true);

        prevPts.fromArray(points.toArray());

        // get centroid to pass into calculation
        center = getCentroid(prevPts.toArray());


        return rgb;
    }
}

