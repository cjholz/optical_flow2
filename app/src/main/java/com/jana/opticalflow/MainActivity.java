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
    double realx;
    double realy;
    int frameheight = 1080;
    int framewidth = 1440;


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

        //Calculate FOV at walking distance
        double realZ = 12.8444422;
        //Calculate diagonal in walking plane
        double diag = (Math.tan(.658)*realZ)*2;
        //Calculate angle of diagonal with x-axis
        double cornerAng = Math.atan((double)frameheight/framewidth);
        //Calculate x and y dimensions in walking plane
        realx = Math.cos(cornerAng)*diag;
        realy = Math.sin(cornerAng)*diag;
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
        MatOfByte status = new MatOfByte();
        MatOfFloat err = new MatOfFloat();
        if (previousFrame != null) {
            calcOpticalFlowPyrLK(previousFrame, gray, prevPts, nextPts, status, err, new Size(21, 21), 2);

            //Calculate and select maximum displacement
            double maxdis = 0;
            int maxdisind = 0;
            for (int i = 0; i < nextPts.toList().size(); i++) {
                if (status.toList().get(i) == 1) {
                    Point displacement = prevPts.toArray()[i].subtract(nextPts.toArray()[i]);
                    double displacementmag = Math.sqrt(Math.pow(displacement.x, 2) + Math.pow(displacement.y, 2));
                    if (displacementmag > maxdis) {
                        maxdis = displacementmag;
                        maxdisind = i;
                    }
                }
            }
            //Calculate velocity for the maximum displacement
            Point displacement = nextPts.toArray()[maxdisind].subtract(prevPts.toArray()[maxdisind]);
            double displacex = displacement.x;
            double displacey = displacement.y;
            double centerx = nextPts.toArray()[maxdisind].x;
            double centery = nextPts.toArray()[maxdisind].y;
            double xvelp = displacex / timeDiff;
            double yvelp = displacey / timeDiff;

            //Convert pixel velocity to m/s
            double xvel = (realx / rgb.cols()) * xvelp;
            double yvel = (realy / rgb.rows()) * yvelp;

            //Smooth velocity over WINDOW_SIZE
            if (xvels.size() >= 5) {
                xvels.remove(0);
                yvels.remove(0);
            }
            xvels.add(xvel);
            yvels.add(yvel);
            xvel = 0;
            yvel = 0;
            for (int i = 0; i < xvels.size(); i++) {
                xvel += xvels.get(i);
                yvel += yvels.get(i);
            }
            xvel = xvel / 5.;
            yvel = yvel / 5.;

            //Determine if velocity is above the threshold and draw displacement vector and velocity meter
            double magn = Math.sqrt(Math.pow(xvel, 2) + Math.pow(yvel, 2));
            if (magn > 0.05) {
                Imgproc.arrowedLine(rgb, new Point(centerx, centery), new Point(centerx + displacex, centery + displacey), new Scalar(255, 0, 0), 20);
                Imgproc.line(rgb, new Point(rgb.width()/2, rgb.height()-50), new Point(rgb.width()/2 + xvel*100, rgb.height()-50), new Scalar(255, 255, 0), 50);
            } else {
                xvel = 0;
                yvel = 0;
            }
            //Draw velocity values with text
            Imgproc.putText(rgb, "XVEL" + String.valueOf(xvel) + "m/s", new Point(30, 75), Core.FONT_HERSHEY_PLAIN, 5.0, new Scalar(255, 255, 0), Imgproc.LINE_8);
            Imgproc.putText(rgb, "YVEL" + String.valueOf(yvel) + "m/s", new Point(30, 150), Core.FONT_HERSHEY_PLAIN, 5.0, new Scalar(255, 255, 0), Imgproc.LINE_8);
        }

        previousFrame = inputFrame.gray();
        // MatofPoint object for import features (corners)
        MatOfPoint points = new MatOfPoint();

        // Generating an object of the most important features (we are using corners of objects)
        Imgproc.goodFeaturesToTrack(previousFrame, points, 100, .1, 21, new Mat(), 3, true);

        prevPts.fromArray(points.toArray());

        // get centroid to pass into calculation
        //center = getCentroid(prevPts.toArray());


        return rgb;
    }
}

